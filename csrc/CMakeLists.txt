
find_package(Torch REQUIRED)
message("Torch CXX FLAGS: ${TORCH_CXX_FLAGS}")
message("Torch LIBS: ${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(USE_STATIC_MKL OFF)
find_package(MKL REQUIRED)
message("MKL found: ${MKL_FOUND}, 
        version: ${MKL_VERSION}, 
        include dir: ${MKL_INCLUDE_DIR}, 
        libs: ${MKL_LIBRARIES},
        omp type: ${MKL_OPENMP_TYPE},
        omp path: ${MKL_OPENMP_LIBRARY}")

set(LIB_SOURCE_FILES my_linear_layer.cpp)

add_executable(hello_torch ${LIB_SOURCE_FILES})

target_link_libraries(hello_torch ${TORCH_LIBRARIES} ${MKL_LIBRARIES})