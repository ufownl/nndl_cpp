aux_source_directory(. NNDL_CHAPTER6_SOURCES)
add_executable(nndl_chapter6 ${NNDL_CHAPTER6_SOURCES})
target_link_libraries(nndl_chapter6 dlib)
