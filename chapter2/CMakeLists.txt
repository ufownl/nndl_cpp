aux_source_directory(. NNDL_CHAPTER2_SOURCES)
add_executable(nndl_chapter2 ${NNDL_CHAPTER2_SOURCES})
target_link_libraries(nndl_chapter2 nndl_utils)
add_dependencies(nndl_chapter2 nndl_utils)
