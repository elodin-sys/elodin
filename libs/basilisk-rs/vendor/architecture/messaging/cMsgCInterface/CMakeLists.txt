SET(LIB_DIR "${CMAKE_SOURCE_DIR}/simulation/architecture/messaging/cMsgCInterface/")

# Determine the name of the parent directory of _GeneralModuleFiles (i.e. dynamics, environment, etc)
get_filename_component(PARENT_DIR ${LIB_DIR} DIRECTORY) # Relative to the source directory (e.g. simulation/power)
string(FIND ${PARENT_DIR} "/" DIR_NAME_IDX REVERSE)
math(EXPR DIR_NAME_IDX "${DIR_NAME_IDX} + 1")# "simulation/"
string(SUBSTRING ${PARENT_DIR} ${DIR_NAME_IDX} -1 PARENT_DIR_NAME) # Should be "power"

# Set the library name (ex. dynamicsLib, environmentLib, etc)
set(LIB_NAME "cMsgCInterface")

# Grab the library source files
file(GLOB C_FILES
        "${CMAKE_SOURCE_DIR}/../dist3/autoSource/cMsgCInterface/*.cpp"
        "${CMAKE_SOURCE_DIR}/../dist3/autoSource/cMsgCInterface/*.c"
        "${CMAKE_SOURCE_DIR}/../dist3/autoSource/cMsgCInterface/*.h"
    )

# Add Target
add_library(${LIB_NAME} STATIC ${C_FILES} ${BSK_FWK_FILES})
if(NOT WIN32)
	target_compile_options(${LIB_NAME} PUBLIC "-fPIC")
endif()

# Link all necessary libraries
#target_link_libraries(${LIB_NAME} ArchitectureUtilities)
target_link_libraries(${LIB_NAME} ${PYTHON_LIBRARIES})
#target_link_libraries(${LIB_NAME} Eigen3::Eigen3)

# define build location, IDE generation specifications
set_target_properties(${LIB_NAME} PROPERTIES FOLDER ${PARENT_DIR})
set_target_properties(${LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(${LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(${LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Basilisk")

# Define the location for the executable part of the library (.dll) files (Windows Only)
# https://cmake.org/cmake/help/v3.17/manual/cmake-buildsystem.7.html#output-artifacts for details
set_target_properties(${LIB_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(${LIB_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(${LIB_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Basilisk")

# Define the location of the import library file (.lib) needed for the .dll (Windows Only)
set_target_properties(${LIB_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(${LIB_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Basilisk")
set_target_properties(${LIB_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Basilisk")
