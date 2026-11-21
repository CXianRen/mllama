# GLM5-Next CPU CLI

Run a local GLM5-Next GGUF on x86 with `llama-cli`.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Run

Generate one reply using the chat template embedded in the GGUF.

```sh
./build/bin/llama-cli -m /path/to/glm5-next.gguf -p "Explain virtual memory." -n 128 --single-turn
```

Use `--no-conversation` for raw prompt completion. Omit `-p` to start an interactive conversation with an instruction GGUF. Use `-c` to set the context size, `-t` to set CPU threads, or `-o` to write generated text to a file. Run `llama-cli --help` for the full list of options.

## Code layout

| Path | Responsibility |
| --- | --- |
| `tools/cli/cli.cpp` | CLI entry point and generation loop |
| `common/` | Arguments, prompt rendering, tokenization helpers, and sampler configuration |
| `src/` | GGUF model loading, KV and recurrent state, and graph execution |
| `src/models/glm5-next.cpp` | GLM5-Next graph |
| `ggml/src/ggml-cpu/` | CPU backend with x86 kernels |
| `tests/` | Optional CPU, GGUF, parser, and CLI regression tests |

The test build creates a small synthetic GLM5-Next GGUF locally.

```sh
cmake -S . -B build-tests -DLLAMA_BUILD_TESTS=ON
cmake --build build-tests --parallel
ctest --test-dir build-tests --output-on-failure
```
