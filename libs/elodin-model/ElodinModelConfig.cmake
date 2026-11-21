# Imports shared and static passive-model targets from the package laid out around this file.
if(NOT TARGET ElodinModel::model)
  get_filename_component(_elodin_model_root "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
  if(APPLE)
    set(_elodin_model_shared "libelodin_model.dylib")
    set(_elodin_model_soname "@rpath/libelodin_model.dylib")
  else()
    set(_elodin_model_shared "libelodin_model.so")
    set(_elodin_model_soname "libelodin_model.so")
  endif()
  set(_elodin_model_static "libelodin_model.a")

  # Keep the historical name as the concrete imported target: CMake's
  # install(IMPORTED_RUNTIME_ARTIFACTS) rejects aliases.
  add_library(ElodinModel::model SHARED IMPORTED)
  set_target_properties(ElodinModel::model PROPERTIES
    IMPORTED_LOCATION "${_elodin_model_root}/lib/${_elodin_model_shared}"
    IMPORTED_SONAME "${_elodin_model_soname}"
    INTERFACE_INCLUDE_DIRECTORIES "${_elodin_model_root}/include")
  add_library(ElodinModel::shared ALIAS ElodinModel::model)

  find_package(Threads REQUIRED)
  add_library(ElodinModel::static STATIC IMPORTED)
  set_target_properties(ElodinModel::static PROPERTIES
    IMPORTED_LOCATION "${_elodin_model_root}/lib/${_elodin_model_static}"
    INTERFACE_INCLUDE_DIRECTORIES "${_elodin_model_root}/include"
    INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m")

  unset(_elodin_model_root)
  unset(_elodin_model_shared)
  unset(_elodin_model_static)
  unset(_elodin_model_soname)
endif()
