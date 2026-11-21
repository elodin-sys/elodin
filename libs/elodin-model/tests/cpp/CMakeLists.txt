cmake_minimum_required(VERSION 3.21)
project(elodin_model_cpp_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ElodinModel CONFIG REQUIRED)

add_executable(passive_shared passive.cpp)
target_link_libraries(passive_shared PRIVATE ElodinModel::shared)

add_executable(passive_static passive.cpp)
target_link_libraries(passive_static PRIVATE ElodinModel::static)

add_executable(interface_smoke interface_smoke.cpp)
target_link_libraries(interface_smoke PRIVATE ElodinModel::shared)

add_library(test_step_hook SHARED test_step_hook.cpp)
target_include_directories(test_step_hook PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../include")
add_executable(hook_smoke hook_smoke.cpp)
target_link_libraries(hook_smoke PRIVATE ElodinModel::shared)
