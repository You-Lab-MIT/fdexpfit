cmake_minimum_required(VERSION 3.22.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(lmfit)
enable_language(CUDA)

# Pybind11
# set(pybind11_DIR ".../share/cmake/pybind11/") # hint for pybind11 location
find_package(pybind11 REQUIRED)

# LibTorch
# set(CMAKE_PREFIX_PATH ".../pytorch/lib/python3.13/site-packages/torch/share/cmake") # hint for pytorch location
find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Generate Python module
pybind11_add_module(lmfit "lmfit.cpp")

# Compile options for optimization
set(COMPILE_FLAGS -Wall)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -fvisibility=hidden -flto -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_compile_options(lmfit PUBLIC ${COMPILE_FLAGS})

include_directories("${TORCH_INCLUDE_DIRS}")
target_link_libraries(lmfit PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(lmfit PRIVATE ${TORCH_PYTHON_LIBRARY})

# Create the output in the source directory
set_target_properties(lmfit PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
