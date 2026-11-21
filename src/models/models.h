#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

#include <cmath>

class llama_memory_hybrid_idx_context;

// ref: https://github.com/ggml-org/llama.cpp/pull/28068
static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};

struct llama_model_glm5_next : public llama_model_base {
    llama_model_glm5_next(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    uint32_t graph_nodes_per_tensor() const override { return 64; }

    // k-pool indexer inputs on top of the generic hybrid input
    class llm_graph_input_kpool;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);

        // build the helpers without the trunk, so graph_mtp can reuse them
        struct no_build {};
        graph(const llm_graph_params & params, no_build);

        // collapse the hc streams with per-stream weights
        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * weights,
                int il) const;

        // mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
        ggml_tensor * build_hc_mean(ggml_tensor * x) const;

        // all outputs of the hc mix builder: cur is the collapsed input, post / comb feed build_hc_post
        struct hc_pre {
            ggml_tensor * cur;
            ggml_tensor * post;
            ggml_tensor * comb;
        };

        hc_pre build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_sinkhorn(
                ggml_tensor * comb,
                int il) const;

        llm_graph_input_kpool * build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb);

        ggml_tensor * build_kda_layer(ggml_tensor * cur, const llama_layer & layer,
                                      llm_graph_input_rs * inp_rs,
                                      int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                      int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il);

        // causal conv1d over one of Q/K/V, qkv selects which third of the conv state row is used
        ggml_tensor * build_kda_conv1d(llm_graph_input_rs * inp_rs, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                       int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                       int64_t d_conv, int64_t head_dim, int64_t n_head_kda, int64_t n_seq_tokens, int64_t n_seqs);

        ggml_tensor * build_kpool_select(ggml_tensor * cur, ggml_tensor * qr, const llama_layer & layer,
                                         const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_kpool * inp_kpool, int il);

        // attention output (cur) and the selection to carry to the next shared layer (sel)
        struct dsa_out {
            ggml_tensor * cur;
            ggml_tensor * sel;
        };

        dsa_out build_dsa_layer(ggml_tensor * cur, const llama_layer & layer,
                                const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
                                llm_graph_input_kpool * inp_kpool, ggml_tensor * prev_sel, int il);

    };

    // the NextN / MTP draft head: one DSA block appended after the trunk
    struct graph_mtp : public graph {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
