add_executable(llama-cli cli.cpp)
target_link_libraries(llama-cli PRIVATE llama-common)
install(TARGETS llama-cli RUNTIME)
