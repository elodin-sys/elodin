# Imports ElodinModel::model from the package package.py laid out around this file.
if(NOT TARGET ElodinModel::model)
  get_filename_component(_elodin_model_root "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
  if(APPLE)
    set(_elodin_model_file "libelodin_model.dylib")
    set(_elodin_model_soname "@rpath/libelodin_model.dylib")
  else()
    set(_elodin_model_file "libelodin_model.so")
    set(_elodin_model_soname "libelodin_model.so")
  endif()

  add_library(ElodinModel::model SHARED IMPORTED)
  set_target_properties(ElodinModel::model PROPERTIES
    IMPORTED_LOCATION "${_elodin_model_root}/lib/${_elodin_model_file}"
    IMPORTED_SONAME "${_elodin_model_soname}"
    INTERFACE_INCLUDE_DIRECTORIES "${_elodin_model_root}/include")

  unset(_elodin_model_root)
  unset(_elodin_model_file)
  unset(_elodin_model_soname)
endif()
