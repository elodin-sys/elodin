include(usingOpenCV)
if(APPLE)
  link_libraries("-march=haswell -mno-lzcnt")
endif()
