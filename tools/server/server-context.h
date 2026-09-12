#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "json.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>

struct server_context_impl; // private implementation

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    json json_ui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    std::string model_ftype;
};

enum server_state {
    SERVER_STATE_DOWNLOADING,
    SERVER_STATE_LOADING,
    SERVER_STATE_READY,
    SERVER_STATE_SLEEPING,
};

static std::string server_state_to_str(server_state state) {
    switch (state) {
        case SERVER_STATE_DOWNLOADING: return "downloading";
        case SERVER_STATE_LOADING:     return "loading";
        case SERVER_STATE_READY:       return "ready";
        case SERVER_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid server_state");
    }
}

static server_state server_state_from_str(const std::string & str) {
    if (str == "downloading") return SERVER_STATE_DOWNLOADING;
    if (str == "loading")     return SERVER_STATE_LOADING;
    if (str == "ready")       return SERVER_STATE_READY;
    if (str == "sleeping")    return SERVER_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid server_state string");
}

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // level-2 prompt cache, can return nullptr if disabled (--cache-ram 0)
    // note: created only once, so the pointer stays valid across sleeping states
    server_prompt_cache * get_prompt_cache() const;

    // note: must be set before load_model() is called
    void set_state_callback(server_state_callback_t callback);
};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta         = std::make_unique<server_context_meta>(ctx_server.get_meta());
        this->prompt_cache = ctx_server.get_prompt_cache();
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t get_chat_completions;          // list stored
    server_http_context::handler_t get_chat_completion;           // retrieve by id
    server_http_context::handler_t post_chat_completion_update;   // update metadata
    server_http_context::handler_t delete_chat_completion;        // delete stored
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t get_responses_oai;
    server_http_context::handler_t delete_responses_oai;
    server_http_context::handler_t post_responses_cancel_oai;
    server_http_context::handler_t post_responses_compact_oai;
    server_http_context::handler_t get_responses_input_items_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_conversations_oai;
    server_http_context::handler_t get_conversation_oai;
    server_http_context::handler_t post_conversation_update_oai;
    server_http_context::handler_t delete_conversation_oai;
    server_http_context::handler_t post_conversation_items_oai;
    server_http_context::handler_t get_conversation_items_oai;
    server_http_context::handler_t get_conversation_item_oai;
    server_http_context::handler_t delete_conversation_item_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    // level-2 prompt cache, owned by ctx_server, nullptr if disabled (--cache-ram 0)
    // note: read by /props, which must also work while the model is unloaded
    server_prompt_cache * prompt_cache = nullptr;

    const common_params & params;
    server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);

    // cached responses, to be used during sleep
    std::mutex     mutex_cache;
    json           cached_models  = nullptr;
    json           cached_props   = nullptr;
    server_metrics cached_metrics;
    // set when a scrape during sleep already reported the throughput buckets
    bool           should_reset_buckets = false;
    // call right before sleep to update the cached responses
    void update_cached_responses(bool is_sleeping);
};
