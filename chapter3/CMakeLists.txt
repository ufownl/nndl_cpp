aux_source_directory(. NNDL_CHAPTER3_SOURCES)
add_executable(nndl_chapter3 ${NNDL_CHAPTER3_SOURCES})
target_link_libraries(nndl_chapter3 nndl_utils)
add_dependencies(nndl_chapter3 nndl_utils)
