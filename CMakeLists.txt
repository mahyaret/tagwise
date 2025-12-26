cmake_minimum_required(VERSION 3.21)
project(tagwise LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Windows (vcpkg) paths ----
if(WIN32)
    # Change this one line if your vcpkg lives elsewhere:
    set(VCPKG_INSTALLED_DIR "C:/Users/home/source/repos/vcpkg/installed/x64-windows")

    # Help CMake find package configs (OpenCV, nlohmann_json, possibly onnxruntime)
    list(APPEND CMAKE_PREFIX_PATH
        "${VCPKG_INSTALLED_DIR}"
        "${VCPKG_INSTALLED_DIR}/share/opencv4"
        "${VCPKG_INSTALLED_DIR}/share/nlohmann_json"
        "${VCPKG_INSTALLED_DIR}/share/onnxruntime"
        "${VCPKG_INSTALLED_DIR}/share/onnxruntime-gpu"
    )

    # Also expose the lib dir explicitly (as requested)
    link_directories("${VCPKG_INSTALLED_DIR}/lib")
endif()

# ---- Sources ----
file(GLOB SRC_FILES
    "${CMAKE_SOURCE_DIR}/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)
add_executable(tagwise ${SRC_FILES})

# ---- Dependencies from vcpkg prefix ----
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json CONFIG REQUIRED)

# ONNX Runtime is OPTIONAL here. If found, we link and define a flag for your code.
find_package(onnxruntime CONFIG QUIET)

# Includes (fixes opencv2-under-opencv4 layout)
target_include_directories(tagwise PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    $<$<BOOL:${WIN32}>:"${VCPKG_INSTALLED_DIR}/include">            # general headers from vcpkg
    $<$<BOOL:${WIN32}>:"${VCPKG_INSTALLED_DIR}/include/opencv4">    # just in case
)

# Link
target_link_libraries(tagwise PRIVATE
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

if(onnxruntime_FOUND)
    target_link_libraries(tagwise PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(tagwise PRIVATE USE_ONNXRUNTIME=1)
else()
    message(WARNING "onnxruntime not found in ${VCPKG_INSTALLED_DIR}; building WITHOUT model inference")
    # If you later want to hard-wire a manual ORT, uncomment and set paths:
    # add_library(onnxruntime SHARED IMPORTED GLOBAL)
    # set_property(TARGET onnxruntime PROPERTY IMPORTED_IMPLIB "${VCPKG_INSTALLED_DIR}/lib/onnxruntime.lib")
    # set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION  "${VCPKG_INSTALLED_DIR}/bin/onnxruntime.dll")
    # target_include_directories(tagwise PRIVATE "${VCPKG_INSTALLED_DIR}/include")
    # target_link_libraries(tagwise PRIVATE onnxruntime)
endif()

# MSVC niceties
if(MSVC)
  target_compile_options(tagwise PRIVATE /W4 /permissive- /Zc:__cplusplus)
endif()

# Output under build/bin
set_target_properties(tagwise PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
