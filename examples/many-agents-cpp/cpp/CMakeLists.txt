cmake_minimum_required(VERSION 3.21)
project(many_agents_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# A fused multiply-add rounds once, so the flight software's commands would differ from the Rust
# controller's, which rustc never fuses.
add_compile_options(-ffp-contract=off -Wall -Wextra)

add_library(flight_software STATIC flight_software.cpp)
target_include_directories(flight_software PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(many_agents_fsw fsw_main.cpp)
target_link_libraries(many_agents_fsw PRIVATE flight_software)

find_package(ElodinModel CONFIG REQUIRED)
find_package(Threads REQUIRED)
add_executable(many_agents_cpp main.cpp model.cpp exchange.cpp witness.cpp blake3.cpp)
target_link_libraries(many_agents_cpp PRIVATE flight_software ElodinModel::model Threads::Threads)

include(GNUInstallDirs)
if(APPLE)
  set(_origin "@loader_path")
else()
  set(_origin "$ORIGIN")
endif()
set_target_properties(many_agents_cpp PROPERTIES INSTALL_RPATH "${_origin}/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS many_agents_cpp many_agents_fsw RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(IMPORTED_RUNTIME_ARTIFACTS ElodinModel::model LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
