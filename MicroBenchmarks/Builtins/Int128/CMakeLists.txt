set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

llvm_test_run(WORKDIR ${CMAKE_CURRENT_BINARY_DIR})

llvm_test_executable(Builtins main.cpp)

target_link_libraries(Builtins benchmark)
