#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

LLAMA_API std::vector<ggml_tensor *> llama_memory_get_tensors(const llama_memory_t mem);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Questions packed after a shared state in one sequence, contexts without memory only
// 0 (default) is the state, k > 0 the k-th question; full attention layers hide other questions from a question
LLAMA_API bool llama_batch_ext_set_decision_group(struct llama_batch_ext * batch, int32_t idx, int32_t group);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

//
// model/context data extraction
//

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);
