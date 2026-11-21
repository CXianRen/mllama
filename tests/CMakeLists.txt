function(llama_add_test name)
    add_executable(${name} ${name}.cpp ${ARGN})
    target_link_libraries(${name} PRIVATE llama-common)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

llama_add_test(test-sampling)
llama_add_test(test-reasoning-budget)
llama_add_test(test-gguf)
llama_add_test(test-chat-peg-parser peg-parser/simple-tokenize.cpp)

add_executable(test-llama-archs test-llama-archs.cpp)
target_link_libraries(test-llama-archs PRIVATE llama-common)

set(model_dir "${CMAKE_CURRENT_BINARY_DIR}/models")
file(MAKE_DIRECTORY "${model_dir}")
add_test(NAME test-generate-glm5-next COMMAND test-llama-archs -a glm5-next -s 42 -o "${model_dir}")
set_tests_properties(test-generate-glm5-next PROPERTIES FIXTURES_SETUP glm5-next)

add_test(NAME test-cli-glm5-next COMMAND llama-cli
    -m "${model_dir}/glm5-next-moe.gguf"
    -p Hello -n 8 -c 256 -t 2 --single-turn --no-conversation --simple-io --no-display-prompt)
set_tests_properties(test-cli-glm5-next PROPERTIES FIXTURES_REQUIRED glm5-next)

add_test(NAME test-cli-glm5-next-chat COMMAND llama-cli
    -m "${model_dir}/glm5-next-moe.gguf"
    --chat-template-file "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat-templates/GLM-4.7-Flash.jinja"
    -p Hello -n 8 -c 256 -t 2 --single-turn --simple-io --no-display-prompt --reasoning off)
set_tests_properties(test-cli-glm5-next-chat PROPERTIES FIXTURES_REQUIRED glm5-next)
