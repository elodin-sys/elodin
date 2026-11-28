cmake_minimum_required(VERSION 3.21)
project(ball_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ball_cpp main.cpp)

find_package(ElodinModel CONFIG REQUIRED)
target_link_libraries(ball_cpp PRIVATE ElodinModel::shared)

# The installed bin/ and lib/ can move together.
include(GNUInstallDirs)
if(APPLE)
  set(_origin "@loader_path")
else()
  set(_origin "$ORIGIN")
endif()
set_target_properties(ball_cpp PROPERTIES INSTALL_RPATH "${_origin}/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS ball_cpp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(IMPORTED_RUNTIME_ARTIFACTS ElodinModel::shared LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
