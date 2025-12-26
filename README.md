# tagwise

Tagwise is a command-line tool that automatically analyzes images with a CLIP-based ONNX model to find the most relevant keywords from a predefined vocabulary. It then renames each image file with the top predicted tags and its modification timestamp for easier search, organization, and sharing.

## Models and Data
```
cd tagwise
git clone https://huggingface.co/mahyaret/tagwise ./model 
```

## Build on Windows

Prerequisites:
- Visual Studio 2022 (Desktop development with C++)
- CMake
- vcpkg

Install dependencies with vcpkg:
```
git clone https://github.com/microsoft/vcpkg C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg.exe install opencv4 nlohmann-json onnxruntime --triplet x64-windows
```

Configure and build:
```
cd C:\\Users\\home\\source\\repos\\tagwise

cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DVCPKG_INSTALLED_DIR="C:/vcpkg/installed/x64-windows"
cmake --build build --config Release
```

## Build on macOS

Prerequisites (Homebrew):
```
brew install cmake opencv onnxruntime nlohmann-json
```

Build:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## How to run
```
.\tagwise.exe "[path_to_images_folder]" "[path_to_tags.txt]" "[path_to_model]"
```

If your model folder contains `tags.txt`, you can also run:
```
.\tagwise.exe "[path_to_images_folder]" "[path_to_model]"
```

If you see noisy ONNX schema registration logs on startup, `TAGWISE_VERBOSE_INIT=1` disables Tagwise's stderr silencing during ONNXRuntime initialization.

If your image model has dynamic input size, Tagwise defaults to `224x224`; override with `TAGWISE_IMAGE_SIZE=224` or `TAGWISE_IMAGE_SIZE=336x336`.

### Run on macOS

```
./build/bin/tagwise "[path_to_images_folder]" "[path_to_tags.txt]" "[path_to_model]"
```

Shorthand (uses `[path_to_model]/tags.txt`):
```
./build/bin/tagwise "[path_to_images_folder]" "[path_to_model]"
```


## Demo

![Demo](assets/tagwise.gif)
