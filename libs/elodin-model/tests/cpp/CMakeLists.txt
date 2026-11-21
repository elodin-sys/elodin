# Consumer tests of a packaged model: the rooms model of ../fixtures/rooms for the tests with
# known answers, any model for `threads` and `stack_probe`.
#
#     cmake -S . -B build -DElodinModel_DIR=<package> && cmake --build build
#     ctest --test-dir build --output-on-failure
cmake_minimum_required(VERSION 3.21)
project(elodin_model_tests LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The headers must stay clean under the strictest flags a consumer builds with.
add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)

find_package(ElodinModel CONFIG REQUIRED)
enable_testing()

add_executable(c_api c_api.c)
target_link_libraries(c_api PRIVATE ElodinModel::shared)
add_test(NAME c_api COMMAND c_api)

foreach(link shared static)
  add_executable(wrapper_${link} wrapper.cpp)
  target_link_libraries(wrapper_${link} PRIVATE ElodinModel::${link})
  add_test(NAME wrapper_${link} COMMAND wrapper_${link})
endforeach()

add_executable(threads threads.cpp)
target_link_libraries(threads PRIVATE ElodinModel::shared Threads::Threads)
add_test(NAME threads COMMAND threads)
add_test(NAME threads_one_worker COMMAND threads 1)

add_executable(stack_probe stack_probe.cpp)
target_link_libraries(stack_probe PRIVATE ElodinModel::shared Threads::Threads)
add_test(NAME stack_probe COMMAND stack_probe)

# The hook library needs the package's header and nothing else of it.
add_library(test_step_hook SHARED test_step_hook.cpp)
target_include_directories(test_step_hook PRIVATE
  $<TARGET_PROPERTY:ElodinModel::shared,INTERFACE_INCLUDE_DIRECTORIES>)
add_executable(hook hook.cpp)
target_link_libraries(hook PRIVATE ElodinModel::shared)
add_test(NAME hook COMMAND hook $<TARGET_FILE:test_step_hook>)
