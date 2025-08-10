// tagwise.cpp — batch rename JPGs using top-3 CLIP tags + last modified time
// Usage:
//   tagwise.exe "<folder>" "<tags.txt>" [model_dir]
// If model_dir is omitted, models are loaded from the executable's directory.
// Required files (in model_dir or next to EXE):
//   clip_image.onnx
//   clip_text.onnx
// If text model expects INT64 input_ids (+ attention_mask), also include:
//   vocab.json
//   merges.txt

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <regex>
#include <cctype>
#include <filesystem>
#include <chrono>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace fs = std::filesystem;

// ---------- Config ----------
static const char* kImageOnnxName = "clip_image.onnx";
static const char* kTextOnnxName = "clip_text.onnx";
static const char* kVocabJsonName = "vocab.json";   // for INT64 path
static const char* kMergesTxtName = "merges.txt";   // for INT64 path
static const size_t TOPK_PICK = 3;                  // rename uses top-3

// ---------- Small helpers ----------
struct Emb { std::vector<float> v; };

static float cosine(const Emb& a, const Emb& b) {
    const size_t n = (std::min)(a.v.size(), b.v.size());
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) { dot += a.v[i] * b.v[i]; na += a.v[i] * a.v[i]; nb += b.v[i] * b.v[i]; }
    return float(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

struct IoNames {
    std::vector<Ort::AllocatedStringPtr> holders;
    std::vector<const char*> ptrs;
};
static IoNames get_input_names(Ort::Session& s, Ort::AllocatorWithDefaultOptions& alloc) {
    IoNames r; const size_t n = s.GetInputCount();
    r.holders.reserve(n); r.ptrs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        r.holders.emplace_back(s.GetInputNameAllocated(i, alloc));
        r.ptrs.emplace_back(r.holders.back().get());
    }
    return r;
}
static IoNames get_output_names(Ort::Session& s, Ort::AllocatorWithDefaultOptions& alloc) {
    IoNames r; const size_t n = s.GetOutputCount();
    r.holders.reserve(n); r.ptrs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        r.holders.emplace_back(s.GetOutputNameAllocated(i, alloc));
        r.ptrs.emplace_back(r.holders.back().get());
    }
    return r;
}

static std::pair<int, int> get_hw_from_model(Ort::Session& s) {
    auto info = s.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape(); // [1,3,H,W]
    if (shape.size() != 4) throw std::runtime_error("Unexpected image input rank (expected NCHW).");
    return { int(shape[2]), int(shape[3]) };
}

// CLIP preprocess with dynamic H,W
static cv::Mat preprocess_clip(const cv::Mat& bgr, int H, int W) {
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    const int iw = rgb.cols, ih = rgb.rows;
    const float scale = float((std::min)(W, H)) / float((std::min)(iw, ih));
    cv::Mat resized;
    cv::resize(rgb, resized,
        cv::Size(int(std::round(iw * scale)), int(std::round(ih * scale))),
        0, 0, cv::INTER_AREA);
    const int x = (resized.cols - W) / 2, y = (resized.rows - H) / 2;
    cv::Rect roi((std::max)(0, x), (std::max)(0, y), W, H);
    cv::Mat crop = resized(roi).clone();

    crop.convertTo(crop, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3); cv::split(crop, ch);
    const float mean[3] = { 0.48145466f,0.4578275f,0.40821073f };
    const float stdv[3] = { 0.26862954f,0.26130258f,0.27577711f };
    for (int c = 0; c < 3; ++c) ch[c] = (ch[c] - mean[c]) / stdv[c];
    cv::Mat chw; cv::vconcat(ch, chw); // 3xHxW contiguous float32
    return chw;
}

static Emb run_clip_image(Ort::Session& sess, const cv::Mat& bgr) {
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_names = get_input_names(sess, alloc);
    auto out_names = get_output_names(sess, alloc);

    auto [H, W] = get_hw_from_model(sess);
    cv::Mat chw = preprocess_clip(bgr, H, W);

    std::vector<int64_t> shape = { 1,3,H,W };
    std::vector<float> data(size_t(3) * H * W);
    std::memcpy(data.data(), chw.ptr<float>(0), data.size() * sizeof(float));

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(),
        shape.data(), shape.size());

    auto outputs = sess.Run(Ort::RunOptions{ nullptr },
        in_names.ptrs.data(), &input, 1,
        out_names.ptrs.data(), 1);

    auto& out = outputs.front();
    float* p = out.GetTensorMutableData<float>();
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape();
    size_t dim = 1; for (auto d : shp) dim *= size_t(d);

    Emb e; e.v.assign(p, p + dim);
    double n = 0; for (float v : e.v) n += double(v) * v; n = std::sqrt(n) + 1e-12;
    for (auto& v : e.v) v = float(v / n);
    return e;
}

// =================== Minimal CLIP Byte-Level BPE Tokenizer ===================
struct ClipBPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    int bos_id = -1, eos_id = -1, pad_id = 0; // common CLIP defaults: pad often 0, eos ~49407

    static std::vector<std::string> regex_split(const std::string& text) {
        // Approx GPT-2 pattern, ok for prompts like "a photo of {tag}"
        std::regex re(R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+)");
        std::vector<std::string> out;
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end = std::sregex_iterator();
        for (auto i = begin; i != end; ++i) {
            std::string tok = (*i).str();
            if (!tok.empty()) out.push_back(tok);
        }
        return out;
    }
    static std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& tokens) {
        std::set<std::pair<std::string, std::string>> pairs;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) pairs.emplace(tokens[i], tokens[i + 1]);
        return pairs;
    }
    std::vector<std::string> bpe(const std::string& token) const {
        if (token.size() == 0) return {};
        std::vector<std::string> word;
        for (size_t i = 0; i < token.size();) {
            unsigned char c = (unsigned char)token[i];
            size_t adv = 1;
            if ((c & 0xE0) == 0xC0) adv = 2;
            else if ((c & 0xF0) == 0xE0) adv = 3;
            else if ((c & 0xF8) == 0xF0) adv = 4;
            word.emplace_back(token.substr(i, adv));
            i += adv;
        }
        if (word.size() == 1) return word;

        auto pairs = get_pairs(word);
        while (true) {
            int min_rank = INT_MAX;
            std::pair<std::string, std::string> bigram;
            for (auto& p : pairs) {
                auto it = bpe_ranks.find(p);
                if (it != bpe_ranks.end() && it->second < min_rank) {
                    min_rank = it->second; bigram = p;
                }
            }
            if (min_rank == INT_MAX) break;

            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size();) {
                if (i + 1 < word.size() && word[i] == bigram.first && word[i + 1] == bigram.second) {
                    new_word.push_back(word[i] + word[i + 1]);
                    i += 2;
                }
                else { new_word.push_back(word[i]); i += 1; }
            }
            word.swap(new_word);
            if (word.size() == 1) break;
            pairs = get_pairs(word);
        }
        return word;
    }
    void load(const fs::path& vocab_json, const fs::path& merges_txt) {
        // vocab
        std::ifstream vf(vocab_json);
        if (!vf) throw std::runtime_error("Cannot open vocab.json");
        json j; vf >> j;
        for (auto it = j.begin(); it != j.end(); ++it) vocab[it.key()] = it.value().get<int>();
        if (vocab.count("<|startoftext|>")) bos_id = vocab["<|startoftext|>"];
        if (vocab.count("<|endoftext|>"))   eos_id = vocab["<|endoftext|>"];
        if (vocab.count("<|pad|>"))         pad_id = vocab["<|pad|>"];

        // merges
        std::ifstream mf(merges_txt);
        if (!mf) throw std::runtime_error("Cannot open merges.txt");
        std::string line; int rank = 0;
        std::getline(mf, line);
        if (line.find('#') == std::string::npos) {
            std::istringstream iss(line); std::string a, b; if (iss >> a >> b) bpe_ranks[{a, b}] = rank++;
        }
        while (std::getline(mf, line)) {
            std::istringstream iss(line); std::string a, b; if (iss >> a >> b) bpe_ranks[{a, b}] = rank++;
        }
    }
    void encode(const std::string& prompt, int max_len,
        std::vector<int64_t>& ids, std::vector<int64_t>& mask) const {
        std::vector<std::string> pieces;
        for (auto& w : regex_split(prompt)) {
            for (auto& bp : bpe(w)) pieces.push_back(bp);
        }
        std::vector<int64_t> tmp;
        tmp.reserve(pieces.size() + 2);
        if (bos_id >= 0) tmp.push_back(bos_id);
        for (auto& p : pieces) {
            auto it = vocab.find(p);
            if (it != vocab.end()) tmp.push_back(it->second);
        }
        if (eos_id >= 0) tmp.push_back(eos_id);

        ids.assign(max_len, pad_id);
        mask.assign(max_len, 0);
        size_t L = (std::min)(size_t(max_len), tmp.size());
        for (size_t i = 0; i < L; ++i) { ids[i] = tmp[i]; mask[i] = 1; }
    }
};
// ===========================================================================

// STRING-input text model
static Emb run_clip_text_string(Ort::Session& sess, const std::string& prompt) {
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_names = get_input_names(sess, alloc);
    auto out_names = get_output_names(sess, alloc);

    auto ti = sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
        throw std::runtime_error("Text model is not STRING-input.");

    std::vector<int64_t> shape = { 1 };
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::Value input = Ort::Value::CreateTensor(ort_alloc, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    const char* s = prompt.c_str();
    input.FillStringTensor(&s, 1);

    auto outputs = sess.Run(Ort::RunOptions{ nullptr },
        in_names.ptrs.data(), &input, 1,
        out_names.ptrs.data(), 1);

    auto& out = outputs.front();
    float* p = out.GetTensorMutableData<float>();
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape();

    Emb e;
    if (shp.size() == 2) {
        size_t dim = size_t(shp[1]); e.v.assign(p, p + dim);
    }
    else if (shp.size() == 3) {
        int64_t seq = shp[1], dim = shp[2];
        e.v.assign(size_t(dim), 0.f);
        for (int64_t t = 0; t < seq; ++t) {
            const float* row = p + t * dim;
            for (int64_t j = 0; j < dim; ++j) e.v[size_t(j)] += row[j];
        }
        for (float& v : e.v) v = v / float(seq);
    }
    else throw std::runtime_error("Unexpected text output rank.");

    double n = 0; for (float v : e.v) n += double(v) * v; n = std::sqrt(n) + 1e-12;
    for (auto& v : e.v) v = float(v / n);
    return e;
}

// INT64-input text model
static Emb run_clip_text_ids(Ort::Session& sess,
    const std::vector<int64_t>& ids,
    const std::vector<int64_t>& mask) {
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_names = get_input_names(sess, alloc);
    auto out_names = get_output_names(sess, alloc);

    std::vector<int64_t> shape = { 1, (int64_t)ids.size() };
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value ids_t = Ort::Value::CreateTensor<int64_t>(mem,
        const_cast<int64_t*>(ids.data()), ids.size(), shape.data(), shape.size());
    Ort::Value mask_t = Ort::Value::CreateTensor<int64_t>(mem,
        const_cast<int64_t*>(mask.data()), mask.size(), shape.data(), shape.size());

    const size_t n_inputs = (size_t)sess.GetInputCount();
    std::vector<const char*> in_ptrs;
    std::vector<Ort::Value>  in_vals;
    in_ptrs.push_back(in_names.ptrs[0]); in_vals.push_back(std::move(ids_t));
    if (n_inputs >= 2) { in_ptrs.push_back(in_names.ptrs[1]); in_vals.push_back(std::move(mask_t)); }

    auto outputs = sess.Run(Ort::RunOptions{ nullptr },
        in_ptrs.data(), in_vals.data(), (size_t)in_vals.size(),
        out_names.ptrs.data(), 1);

    auto& out = outputs.front();
    float* p = out.GetTensorMutableData<float>();
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape();

    Emb e;
    if (shp.size() == 2) {
        size_t dim = size_t(shp[1]); e.v.assign(p, p + dim);
    }
    else if (shp.size() == 3) {
        int64_t seq = shp[1], dim = shp[2];
        e.v.assign(size_t(dim), 0.f);
        for (int64_t t = 0; t < seq; ++t) {
            const float* row = p + t * dim;
            for (int64_t j = 0; j < dim; ++j) e.v[size_t(j)] += row[j];
        }
        for (float& v : e.v) v = v / float(seq);
    }
    else throw std::runtime_error("Unexpected text output rank.");

    double n = 0; for (float v : e.v) n += double(v) * v; n = std::sqrt(n) + 1e-12;
    for (auto& v : e.v) v = float(v / n);
    return e;
}

#ifdef _WIN32
static std::wstring wpath(const fs::path& p) { return p.wstring(); }
#endif

// ---------- Tag loading ----------
static std::vector<std::string> load_tags(const std::string& path) {
    std::vector<std::string> tags;
    std::ifstream f(path);
    std::string s;
    while (std::getline(f, s)) {
        // trim
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || std::isspace((unsigned char)s.back()))) s.pop_back();
        size_t i = 0; while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
        if (i > 0) s = s.substr(i);
        if (!s.empty()) tags.push_back(s);
    }
    return tags;
}

// ---------- Sanitize tag for filename ----------
static std::string clean_tag_for_filename(std::string t) {
    for (auto& c : t) {
        if (c == ' ' || c == '-') c = '_';
        else c = (char)std::tolower((unsigned char)c);
    }
    std::string out; out.reserve(t.size());
    for (char c : t) {
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') out.push_back(c);
    }
    std::string out2;
    bool prev_us = false;
    for (char c : out) {
        if (c == '_') { if (!prev_us) { out2.push_back(c); prev_us = true; } }
        else { out2.push_back(c); prev_us = false; }
    }
    if (!out2.empty() && out2.front() == '_') out2.erase(out2.begin());
    if (!out2.empty() && out2.back() == '_') out2.pop_back();
    return out2.empty() ? std::string("tag") : out2;
}

// ---------- File modified time (YYYYMMDD-HHMMSS) ----------
static std::string last_modified_YmdHMS(const fs::path& p) {
    auto ftime = fs::last_write_time(p);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    std::time_t tt = std::chrono::system_clock::to_time_t(sctp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    try {
        if (argc < 3 || argc > 4) {
            std::fprintf(stderr, "Usage: %s <folder> <tags.txt> [model_dir]\n", argv[0]);
            return 1;
        }
        fs::path folder = fs::path(argv[1]);
        std::string tags_path = argv[2];

        // Determine model dir: argument or executable dir
        fs::path model_dir;
        if (argc == 4) {
            model_dir = fs::path(argv[3]);
        }
        else {
            // exe directory
#ifdef _WIN32
            wchar_t buf[MAX_PATH]; GetModuleFileNameW(NULL, buf, MAX_PATH);
            model_dir = fs::path(buf).parent_path();
#else
            model_dir = fs::current_path();
#endif
        }

        if (!fs::exists(folder) || !fs::is_directory(folder)) {
            std::fprintf(stderr, "Folder not found or not a directory: %s\n", folder.string().c_str());
            return 2;
        }

        auto tags = load_tags(tags_path);
        if (tags.empty()) { std::fprintf(stderr, "No tags in %s\n", tags_path.c_str()); return 3; }

        // ORT env + sessions
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tagwise");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(int((std::max)(1u, std::thread::hardware_concurrency())));
        // Enable CUDA if you installed onnxruntime-gpu:
        // OrtCUDAProviderOptions cuda_opts; so.AppendExecutionProvider_CUDA(cuda_opts);

        fs::path img_model = model_dir / kImageOnnxName;
        fs::path txt_model = model_dir / kTextOnnxName;

#ifdef _WIN32
        Ort::Session img_sess(env, wpath(img_model).c_str(), so);
        Ort::Session txt_sess(env, wpath(txt_model).c_str(), so);
#else
        Ort::Session img_sess(env, img_model.string().c_str(), so);
        Ort::Session txt_sess(env, txt_model.string().c_str(), so);
#endif

        // Decide STRING vs INT64 for text model
        auto ti0 = txt_sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        bool text_is_string = (ti0.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

        // If INT64, load tokenizer files
        ClipBPETokenizer tok;
        fs::path vocab_path = model_dir / kVocabJsonName;
        fs::path merges_path = model_dir / kMergesTxtName;
        if (!text_is_string) {
            tok.load(vocab_path, merges_path); // make sure these are CLIP's files (not GPT-2)
        }

        size_t processed = 0, renamed = 0;

        for (const auto& entry : fs::directory_iterator(folder)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
            if (ext != ".jpg" && ext != ".jpeg") continue;

            const fs::path img_path = entry.path();

            // Load image
            cv::Mat bgr = cv::imread(img_path.string());
            if (bgr.empty()) {
                std::fprintf(stderr, "Failed to read image: %s\n", img_path.string().c_str());
                continue;
            }

            // Image embedding
            Emb img_emb = run_clip_image(img_sess, bgr);

            // Score tags
            struct Sc { float s; std::string tag; };
            std::vector<Sc> scored; scored.reserve(tags.size());
            if (text_is_string) {
                for (const auto& t : tags) {
                    std::string prompt = "a photo of " + t;
                    Emb txt_emb = run_clip_text_string(txt_sess, prompt);
                    scored.push_back({ cosine(img_emb, txt_emb), t });
                }
            }
            else {
                for (const auto& t : tags) {
                    const std::string prompt = "a photo of " + t;
                    std::vector<int64_t> ids, mask;
                    tok.encode(prompt, 77, ids, mask);
                    Emb txt_emb = run_clip_text_ids(txt_sess, ids, mask);
                    scored.push_back({ cosine(img_emb, txt_emb), t });
                }
            }

            // Pick top-3
            const size_t K = std::min<size_t>(TOPK_PICK, scored.size());
            std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
                [](const Sc& a, const Sc& b) { return a.s > b.s; });
            std::vector<std::string> top;
            for (size_t i = 0; i < K; ++i) {
                std::string h = clean_tag_for_filename(scored[i].tag);
                if (!h.empty()) top.push_back(h);
            }
            while (top.size() < 3) top.push_back("tag");

            // Timestamp from last modified
            std::string ts = last_modified_YmdHMS(img_path);

            // Build new name
            std::ostringstream base;
            base << top[0] << "_" << top[1] << "_" << top[2] << "_" << ts;
            fs::path new_path = img_path.parent_path() / (base.str() + img_path.extension().string());

            // Avoid overwrite: if exists, add numeric suffix
            int suffix = 1;
            while (fs::exists(new_path) && new_path != img_path) {
                std::ostringstream b2;
                b2 << base.str() << "_" << suffix++;
                new_path = img_path.parent_path() / (b2.str() + img_path.extension().string());
            }

            try {
                fs::rename(img_path, new_path);
                std::printf("Renamed: %s -> %s\n",
                    img_path.filename().string().c_str(),
                    new_path.filename().string().c_str());
                ++renamed;
            }
            catch (const std::exception& e) {
                std::fprintf(stderr, "Rename failed for %s: %s\n",
                    img_path.string().c_str(), e.what());
            }

            ++processed;
        }

        std::printf("Done. Processed: %zu, Renamed: %zu\n", processed, renamed);
        return 0;

    }
    catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[ONNXRuntime] %s\n", e.what());
        return 10;
    }
    catch (const cv::Exception& e) {
        std::fprintf(stderr, "[OpenCV] %s\n", e.what());
        return 11;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "[std] %s\n", e.what());
        return 12;
    }
}
