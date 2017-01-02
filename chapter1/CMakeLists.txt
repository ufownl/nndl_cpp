aux_source_directory(. NNDL_CHAPTER1_SOURCES)
add_executable(nndl_chapter1 ${NNDL_CHAPTER1_SOURCES})
target_link_libraries(nndl_chapter1 nndl_utils)
add_dependencies(nndl_chapter1 nndl_utils)
