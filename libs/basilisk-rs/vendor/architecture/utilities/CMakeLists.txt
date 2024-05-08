cmake_minimum_required(VERSION 3.14)

file(GLOB basilisk_src "*.h" "*.cpp" "*.c" "*.i")

file(GLOB moduleId "moduleIdGenerator/*.h" "moduleIdGenerator/*.cpp" "moduleIdGenerator/*.i")

add_library(ArchitectureUtilities STATIC ${basilisk_src})
add_library(ModuleIdGenerator SHARED ${moduleId})
string(LENGTH ${CMAKE_SOURCE_DIR} DIR_NAME_START)
math(EXPR DIR_NAME_START "${DIR_NAME_START} + 1")
string(SUBSTRING ${CMAKE_CURRENT_SOURCE_DIR} ${DIR_NAME_START} -1 DIR_NAME_STRING)
set_target_properties(ArchitectureUtilities PROPERTIES FOLDER "${DIR_NAME_STRING}")
set_target_properties(ModuleIdGenerator PROPERTIES FOLDER "${DIR_NAME_STRING}")

if(NOT WIN32)
  target_compile_options(ArchitectureUtilities PUBLIC "-fPIC")
  target_compile_options(ModuleIdGenerator PUBLIC "-fPIC")
endif()

set_target_properties(ArchitectureUtilities PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(ArchitectureUtilities PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(ArchitectureUtilities PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Basilisk")

set_target_properties(ModuleIdGenerator PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(ModuleIdGenerator PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(ModuleIdGenerator PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Basilisk")

if(WIN32)
  add_custom_command(
    TARGET ModuleIdGenerator
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ModuleIdGenerator> "${CMAKE_BINARY_DIR}/Basilisk/")
endif()
