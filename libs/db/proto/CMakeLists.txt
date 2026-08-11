cmake_minimum_required(VERSION 3.28)
project(elodin-db-protos VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED CONFIG)
find_package(gRPC REQUIRED CONFIG)
find_package(Threads REQUIRED)

include(GNUInstallDirs)

set(TARGET_NAME elodin-db-protos)
set(PROTO_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/elodin/db/v1/ingest.proto")

add_library(${TARGET_NAME} SHARED ${PROTO_FILES})

protobuf_generate(
    TARGET ${TARGET_NAME}
    LANGUAGE cpp
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    GENERATE_EXTENSIONS .pb.h .pb.cc)

protobuf_generate(
    TARGET ${TARGET_NAME}
    LANGUAGE grpc
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++)

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS ${TARGET_NAME}
    EXPORT elodin-db-protos-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/elodin
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
      PATTERN "*.pb.h"
      PATTERN "*.grpc.pb.h")

install(EXPORT elodin-db-protos-targets
    FILE elodin-db-protos-targets.cmake
    NAMESPACE elodin-db-protos::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elodin-db-protos)

include(CMakePackageConfigHelpers)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/elodin-db-protos-config.cmake
[=[
include(CMakeFindDependencyMacro)

find_dependency(Protobuf CONFIG)
find_dependency(gRPC CONFIG)
find_dependency(Threads)

include("${CMAKE_CURRENT_LIST_DIR}/elodin-db-protos-targets.cmake")
]=])

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/elodin-db-protos-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/elodin-db-protos-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/elodin-db-protos-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elodin-db-protos)
