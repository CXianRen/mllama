#include "arg.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <list>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <system_error>
#include <thread> // for hardware_concurrency
#include <vector>

using json = common_json;
using namespace common_arg_utils;

static std::string read_file(const std::string & fname) {
    std::ifstream file(fname);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

common_arg & common_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

common_arg & common_arg::set_sampling() {
    is_sampling = true;
    return *this;
}

bool common_arg::get_value_from_env(std::string & output) const {
    if (env == nullptr) return false;
    if (!args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        char * neg_value = std::getenv(neg_env.c_str());
        if (neg_value) {
            output = "0"; // falsey
            return true;
        }
    }
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool common_arg::has_value_from_env() const {
    if (env != nullptr && !args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        if (std::getenv(neg_env.c_str())) {
            return true;
        }
    }
    return env != nullptr && std::getenv(env);
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string line;
    auto add_line = [&](const std::string& l) {
        if (l.length() <= max_char_per_line) {
            result.push_back(l);
        } else {
            std::istringstream line_stream(l);
            std::string word, current_line;
            while (line_stream >> word) {
                if (current_line.length() + !current_line.empty() + word.length() > max_char_per_line) {
                    if (!current_line.empty()) result.push_back(current_line);
                    current_line = word;
                } else {
                    current_line += (!current_line.empty() ? " " : "") + word;
                }
            }
            if (!current_line.empty()) result.push_back(current_line);
        }
    };
    while (std::getline(iss, line)) {
        add_line(line);
    }
    return result;
}

std::string common_arg::to_string() const {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    auto all_args = get_args(); // also contains args_neg
    for (const auto & arg : all_args) {
        if (arg == all_args.front()) {
            if (all_args.size() == 1) {
                ss << arg;
            } else {
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
            }
        } else {
            ss << arg << (arg != all_args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

std::vector<std::string> common_arg::get_args() const {
    std::vector<std::string> result;
    for (const auto & arg : args) {
        result.push_back(std::string(arg));
    }
    for (const auto & arg : args_neg) {
        result.push_back(std::string(arg));
    }
    return result;
}

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_F8_E4M3,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::string get_all_kv_cache_types() {
    std::ostringstream msg;
    for (const auto & type : kv_cache_types) {
        msg << ggml_type_name(type) << (&type == &kv_cache_types.back() ? "" : ", ");
    }
    return msg.str();
}

static bool parse_bool_value(const std::string & value) {
    if (is_truthy(value)) {
        return true;
    } else if (is_falsey(value)) {
        return false;
    } else {
        throw std::invalid_argument("invalid boolean value");
    }
}

static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    common_params & params = ctx_arg.params;

    // setup log directly from params.verbosity: see tools/cli/cli.cpp
    common_log_set_verbosity_thold(params.verbosity);

    std::unordered_map<std::string, std::pair<common_arg *, bool>> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = {&opt, /* is_positive */ true};
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = {&opt, /* is_positive */ false};
        }
    }

    // handle environment variables
    for (auto & opt : ctx_arg.options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            try {
                if (opt.handler_void && is_truthy(value)) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_bool) {
                    opt.handler_bool(params, parse_bool_value(value));
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
        }
        auto & tmp = arg_to_options[arg];
        auto opt = *tmp.first;
        bool is_positive = tmp.second;
        if (opt.has_value_from_env()) {
            fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
        }
        try {
            if (opt.handler_void) {
                opt.handler_void(params);
                continue;
            }
            if (opt.handler_bool) {
                opt.handler_bool(params, is_positive);
                continue;
            }

            // arg with single value
            check_arg(i);
            std::string val = argv[++i];
            if (opt.handler_int) {
                opt.handler_int(params, std::stoi(val));
                continue;
            }
            if (opt.handler_string) {
                opt.handler_string(params, val);
                continue;
            }

            // arg with 2 values
            check_arg(i);
            std::string val2 = argv[++i];
            if (opt.handler_str_str) {
                opt.handler_str_str(params, val, val2);
                continue;
            }
        } catch (std::exception & e) {
            throw std::invalid_argument(string_format(
                "error while handling argument \"%s\": %s\n\n"
                "usage:\n%s\n\nto show complete usage, run with -h",
                arg.c_str(), e.what(), opt.to_string().c_str()));
        }
    }

    if (params.n_ctx < 0 || params.n_batch <= 0 || params.n_ubatch <= 0 || params.n_predict < -2) {
        throw std::invalid_argument("invalid context, batch size, or prediction limit");
    }

    postprocess_cpu_params(params.cpuparams,       nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    if (params.sampling.ignore_eos && !common_grammar_value(params.sampling.grammar).empty()) {
        throw std::invalid_argument("error: --ignore-eos is not supported with --grammar or --json-schema\n");
    }

    if (!params.usage && params.model.path.empty()) {
        throw std::invalid_argument("error: --model is required");
    }

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
        for (auto & seq_breaker : params.sampling.dry_sequence_breakers) {
            string_process_escapes(seq_breaker);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    if (!params.chat_template.empty() && !common_chat_verify_template(params.chat_template, params.use_jinja)) {
        throw std::runtime_error(string_format(
            "error: the supplied chat template is not supported: %s%s\n",
            params.chat_template.c_str(),
            params.use_jinja ? "" : "\nnote: llama.cpp was started without --jinja, we only support commonly used templates"
        ));
    }

    // if the preserve_reasoning kwarg was not specified explicitly, enable it by default
    if (!params.default_template_kwargs.count("preserve_reasoning")) {
        params.default_template_kwargs["preserve_reasoning"] = "true";
    }

    return true;
}

static void common_params_print_usage(const common_params_context & ctx_arg) {
    puts("Options");
    for (const auto & opt : ctx_arg.options) {
        if (!opt.is_sampling) {
            printf("%s", opt.to_string().c_str());
        }
    }
    puts("\nSampling options");
    for (const auto & opt : ctx_arg.options) {
        if (opt.is_sampling) {
            printf("%s", opt.to_string().c_str());
        }
    }
}

void common_print_available_devices() {
    constexpr size_t MiB = 1024 * 1024;
    ggml_backend_load_all();

    const size_t n_devices = ggml_backend_dev_count();
    printf("Available devices:\n");
    if (n_devices == 0) {
        printf("  (none)\n");
        return;
    }
    for (size_t i = 0; i < n_devices; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / MiB, free / MiB);
    }
}

bool common_params_parse(int argc, char ** argv, common_params & params, void(*print_usage)(int, char **)) {
    auto ctx_arg = common_params_parser_init(params, print_usage);
    const common_params params_org = ctx_arg.params;

    try {
        if (!common_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        if (ctx_arg.params.usage) {
            common_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            common_log_flush(common_log_main());
            exit(0);
        }
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    } catch (std::exception & ex) {
        fprintf(stderr, "%s\n", ex.what());
        exit(1); // for other exceptions, we exit with status code 1
    }

    return true;
}

static std::string list_builtin_chat_templates() {
    std::vector<const char *> supported_tmpl;
    int32_t res = llama_chat_builtin_templates(nullptr, 0);
    supported_tmpl.resize(res);
    res = llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
    std::ostringstream msg;
    for (auto & tmpl : supported_tmpl) {
        msg << tmpl << (&tmpl == &supported_tmpl.back() ? "" : ", ");
    }
    return msg.str();
}

bool common_arg_utils::is_truthy(const std::string & value) {
    return value == "on" || value == "enabled" || value == "true" || value == "1";
}

bool common_arg_utils::is_falsey(const std::string & value) {
    return value == "off" || value == "disabled" || value == "false" || value == "0";
}

bool common_arg_utils::is_autoy(const std::string & value) {
    return value == "auto" || value == "-1";
}

// Simple CSV parser that handles quoted fields and escaped quotes
// example:
//    input:  value1,"value, with, commas","value with ""escaped"" quotes",value4
//    output: [value1] [value, with, commas] [value with "escaped" quotes] [value4]
static std::vector<std::string> parse_csv_row(const std::string& input) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < input.length(); ++i) {
        char ch = input[i];

        if (ch == '"') {
            if (!in_quotes) {
                // start of quoted field (only valid if at beginning of field)
                if (!field.empty()) {
                    // quote appeared in middle of unquoted field, treat as literal
                    field += '"';
                } else {
                    in_quotes = true; // start
                }
            } else {
                if (i + 1 < input.length() && input[i + 1] == '"') {
                    // escaped quote: ""
                    field += '"';
                    ++i; // skip the next quote
                } else {
                    in_quotes = false; // end
                }
            }
        } else if (ch == ',') {
            if (in_quotes) {
                field += ',';
            } else {
                fields.push_back(std::move(field));
                field.clear();
            }
        } else {
            field += ch;
        }
    }

    // Add the last field
    fields.push_back(std::move(field));

    return fields;
}

common_params_context common_params_parser_init(common_params & params, void(*print_usage)(int, char **)) {
    params.use_color = tty_can_use_colors();
    common_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sampling.samplers) {
        sampler_type_chars += common_sampler_type_to_chr(sampler);
        sampler_type_names += common_sampler_type_to_str(sampler) + ";";
    }
    if (!sampler_type_names.empty()) {
        sampler_type_names.pop_back(); // remove last semicolon
    }

    auto add_opt = [&](common_arg arg) {
        ctx_arg.options.push_back(std::move(arg));
    };

    add_opt(common_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](common_params & params) {
            params.usage = true;
        }
    ));
    add_opt(common_arg(
        {"--version"},
        "show version and build info",
        [](common_params &) {
            llama_print_build_info(llama_version());
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--verbose-prompt"},
        string_format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](common_params & params) {
            params.verbose_prompt = true;
        }
    ));
    add_opt(common_arg(
        {"--display-prompt"},
        {"--no-display-prompt"},
        string_format("whether to print prompt at generation (default: %s)", params.display_prompt ? "true" : "false"),
        [](common_params & params, bool value) {
            params.display_prompt = value;
        }
    ));
    add_opt(common_arg(
        {"-co", "--color"}, "[on|off|auto]",
        "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.use_color = true;
            } else if (is_falsey(value)) {
                params.use_color = false;
            } else if (is_autoy(value)) {
                params.use_color = tty_can_use_colors();
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --color: '%s'\n", value.c_str()));
            }
        }
    ));
    add_opt(common_arg(
        {"-t", "--threads"}, "N",
        string_format("number of CPU threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](common_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(common_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](common_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(common_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](common_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](common_params & params, const std::string & range) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict"}, "<0|1>",
        string_format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](common_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"--prio"}, "N",
        string_format("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n", params.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < GGML_SCHED_PRIO_LOW || prio > GGML_SCHED_PRIO_REALTIME) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll"}, "<0...100>",
        string_format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](common_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](common_params & params, const std::string & range) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(common_arg(
        {"--prio-batch"}, "N",
        string_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](common_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(common_arg(
        {"-c", "--ctx-size"}, "N",
        string_format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](common_params & params, int value) {
            params.n_ctx = value;
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(common_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        string_format(
            "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)",
            params.n_predict),
        [](common_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(common_arg(
        {"-b", "--batch-size"}, "N",
        string_format("logical maximum batch size (default: %d)", params.n_batch),
        [](common_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(common_arg(
        {"-ub", "--ubatch-size"}, "N",
        string_format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](common_params & params, int value) {
            params.n_ubatch = value;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(common_arg(
        {"--keep"}, "N",
        string_format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](common_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(common_arg(
        {"--swa-full"},
        string_format("use full-size SWA cache (default: %s)\n"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", params.swa_full ? "true" : "false"),
        [](common_params & params) {
            params.swa_full = true;
        }
    ).set_env("LLAMA_ARG_SWA_FULL"));
    add_opt(common_arg(
        {"--context-shift"},
        {"--no-context-shift"},
        string_format("whether to use context shift on infinite text generation (default: %s)", params.ctx_shift ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.ctx_shift = value;
        }
    ).set_env("LLAMA_ARG_CONTEXT_SHIFT"));
    add_opt(common_arg({ "-fa", "--flash-attn" }, "[on|off|auto]",
                       string_format("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
                                     llama_flash_attn_type_name(params.flash_attn_type)),
                       [](common_params & params, const std::string & value) {
                           if (is_truthy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                           } else if (is_falsey(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                           } else if (is_autoy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
                           } else {
                               throw std::runtime_error(
                                   string_format("error: unknown value for --flash-attn: '%s'\n", value.c_str()));
                           }
                       }).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(common_arg(
        {"-p", "--prompt"}, "PROMPT",
        "prompt to start generation with; for system message, use -sys",
        [](common_params & params, const std::string & value) {
            params.prompt = value;
        }
    ));
    add_opt(common_arg(
        {"-sys", "--system-prompt"}, "PROMPT",
        "system prompt to use with model (if applicable, depending on chat template)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = value;
        }
    ));
    add_opt(common_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.prompt = read_file(value);
            // store the external file name in params
            params.prompt_file = value;
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ));
    add_opt(common_arg(
        {"-sysf", "--system-prompt-file"}, "FNAME",
        "a file containing the system prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = read_file(value);
            if (!params.system_prompt.empty() && params.system_prompt.back() == '\n') {
                params.system_prompt.pop_back();
            }
        }
    ));
    add_opt(common_arg(
        {"-e", "--escape"},
        {"--no-escape"},
        string_format("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](common_params & params, bool value) {
            params.escape = value;
        }
    ));
    add_opt(common_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](common_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ));
    add_opt(common_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](common_params & params) {
            params.prompt_cache_all = true;
        }
    ));
    add_opt(common_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](common_params & params) {
            params.prompt_cache_ro = true;
        }
    ));
    add_opt(common_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](common_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ));
    add_opt(common_arg(
        {"-sp", "--special"},
        string_format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](common_params & params) {
            params.special = true;
        }
    ));
    add_opt(common_arg(
        {"-cnv", "--conversation"},
        {"-no-cnv", "--no-conversation"},
        "whether to run in conversation mode:\n"
        "- does not print special tokens and suffix/prefix\n"
        "- interactive mode is also enabled\n"
        "(default: auto enabled if chat template is available)",
        [](common_params & params, bool value) {
            params.conversation_mode = value ? COMMON_CONVERSATION_MODE_ENABLED : COMMON_CONVERSATION_MODE_DISABLED;
        }
    ));
    add_opt(common_arg(
        {"-st", "--single-turn"},
        "run conversation for a single turn only, then exit when done\n"
        "will not be interactive if first turn is predefined with --prompt\n"
        "(default: false)",
        [](common_params & params) {
            params.single_turn = true;
        }
    ));
    add_opt(common_arg(
        {"-i", "--interactive"},
        string_format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](common_params & params) {
            params.interactive = true;
        }
    ));
    add_opt(common_arg(
        {"-if", "--interactive-first"},
        string_format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](common_params & params) {
            params.interactive_first = true;
        }
    ));
    add_opt(common_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](common_params & params) {
            params.multiline_input = true;
        }
    ));
    add_opt(common_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](common_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ));
    add_opt(common_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ));
    add_opt(common_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ));
    add_opt(common_arg(
        {"--warmup"},
        {"--no-warmup"},
        string_format("whether to perform warmup with an empty run (default: %s)", params.warmup ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.warmup = value;
        }
    ));
    add_opt(common_arg(
        {"--samplers"}, "SAMPLERS",
        string_format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](common_params & params, const std::string & value) {
            const auto sampler_names = string_split<std::string>(value, ';');
            params.sampling.samplers = common_sampler_types_from_names(sampler_names);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-s", "--seed"}, "SEED",
        string_format("RNG seed (default: %d, use random seed for %d)", params.sampling.seed, LLAMA_DEFAULT_SEED),
        [](common_params & params, const std::string & value) {
            params.sampling.seed = std::stoul(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--sampler-seq", "--sampling-seq"}, "SEQUENCE",
        string_format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.samplers = common_sampler_types_from_chars(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](common_params & params) {
            params.sampling.ignore_eos = true;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--temp", "--temperature"}, "N",
        string_format("temperature (default: %.2f)", (double)params.sampling.temp),
        [](common_params & params, const std::string & value) {
            params.sampling.temp = std::stof(value);
            params.sampling.temp = std::max(params.sampling.temp, 0.0f);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
        }
    ).set_sampling().set_env("LLAMA_ARG_TEMPERATURE"));
    add_opt(common_arg(
        {"--top-k"}, "N",
        string_format("top-k sampling (default: %d, 0 = disabled)", params.sampling.top_k),
        [](common_params & params, int value) {
            params.sampling.top_k = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
        }
    ).set_sampling().set_env("LLAMA_ARG_TOP_K"));
    add_opt(common_arg(
        {"--top-p"}, "N",
        string_format("top-p sampling (default: %.2f, 1.0 = disabled)", (double)params.sampling.top_p),
        [](common_params & params, const std::string & value) {
            params.sampling.top_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
        }
    ).set_sampling().set_env("LLAMA_ARG_TOP_P"));
    add_opt(common_arg(
        {"--min-p"}, "N",
        string_format("min-p sampling (default: %.2f, 0.0 = disabled)", (double)params.sampling.min_p),
        [](common_params & params, const std::string & value) {
            params.sampling.min_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
        }
    ).set_sampling().set_env("LLAMA_ARG_MIN_P"));
    add_opt(common_arg(
        {"--top-nsigma", "--top-n-sigma"}, "N",
        string_format("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", params.sampling.top_n_sigma),
        [](common_params & params, const std::string & value) {
            params.sampling.top_n_sigma = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-probability"}, "N",
        string_format("xtc probability (default: %.2f, 0.0 = disabled)", (double)params.sampling.xtc_probability),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_probability = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-threshold"}, "N",
        string_format("xtc threshold (default: %.2f, 1.0 = disabled)", (double)params.sampling.xtc_threshold),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_threshold = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--typical", "--typical-p"}, "N",
        string_format("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)", (double)params.sampling.typ_p),
        [](common_params & params, const std::string & value) {
            params.sampling.typ_p = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-last-n"}, "N",
        string_format("last n tokens to consider for penalize (default: %d, 0 = disabled)", params.sampling.penalty_last_n),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::runtime_error(string_format("error: invalid repeat-last-n = %d\n", value));
            }
            params.sampling.penalty_last_n = value;
            params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-penalty"}, "N",
        string_format("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)", (double)params.sampling.penalty_repeat),
        [](common_params & params, const std::string & value) {
            const float penalty_repeat = std::stof(value);
            if (!std::isfinite(penalty_repeat) ||
                penalty_repeat <= 0.0f ||
                !std::isfinite(1.0f/penalty_repeat)) {
                throw std::runtime_error("error: repeat-penalty must be finite and greater than 0\n");
            }
            params.sampling.penalty_repeat = penalty_repeat;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
        }
    ).set_sampling().set_env("LLAMA_ARG_REPEAT_PENALTY"));
    add_opt(common_arg(
        {"--presence-penalty"}, "N",
        string_format("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_present),
        [](common_params & params, const std::string & value) {
            const float penalty_present = std::stof(value);
            if (!std::isfinite(penalty_present)) {
                throw std::runtime_error("error: presence-penalty must be finite\n");
            }
            params.sampling.penalty_present = penalty_present;
        }
    ).set_sampling().set_env("LLAMA_ARG_PRESENCE_PENALTY"));
    add_opt(common_arg(
        {"--frequency-penalty"}, "N",
        string_format("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_freq),
        [](common_params & params, const std::string & value) {
            const float penalty_freq = std::stof(value);
            if (!std::isfinite(penalty_freq)) {
                throw std::runtime_error("error: frequency-penalty must be finite\n");
            }
            params.sampling.penalty_freq = penalty_freq;
        }
    ).set_sampling().set_env("LLAMA_ARG_FREQUENCY_PENALTY"));
    add_opt(common_arg(
        {"--dry-multiplier"}, "N",
        string_format("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)", (double)params.sampling.dry_multiplier),
        [](common_params & params, const std::string & value) {
            params.sampling.dry_multiplier = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-base"}, "N",
        string_format("set DRY sampling base value (default: %.2f)", (double)params.sampling.dry_base),
        [](common_params & params, const std::string & value) {
            float potential_base = std::stof(value);
            if (potential_base >= 1.0f)
            {
                params.sampling.dry_base = potential_base;
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-allowed-length"}, "N",
        string_format("set allowed length for DRY sampling (default: %d)", params.sampling.dry_allowed_length),
        [](common_params & params, int value) {
            params.sampling.dry_allowed_length = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-penalty-last-n"}, "N",
        string_format("set DRY penalty for the last n tokens (default: %d, 0 = disable)", params.sampling.dry_penalty_last_n),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::runtime_error(string_format("error: invalid dry-penalty-last-n = %d\n", value));
            }
            params.sampling.dry_penalty_last_n = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-sequence-breaker"}, "STRING",
        string_format("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers\n",
            params.sampling.dry_sequence_breakers.empty() ? "none" :
            std::accumulate(std::next(params.sampling.dry_sequence_breakers.begin()),
                params.sampling.dry_sequence_breakers.end(),
                std::string("'") + (params.sampling.dry_sequence_breakers[0] == "\n" ? "\\n" : params.sampling.dry_sequence_breakers[0]) + "'",
                [](const std::string& a, const std::string& b) {
                    std::string formatted_b = (b == "\n") ? "\\n" : b;
                    return a + ", '" + formatted_b + "'";
                }).c_str()),
        [](common_params & params, const std::string & value) {
            static bool defaults_cleared = false;

            if (!defaults_cleared) {
                params.sampling.dry_sequence_breakers.clear();
                defaults_cleared = true;
            }

            if (value == "none") {
                params.sampling.dry_sequence_breakers.clear();
            } else {
                params.sampling.dry_sequence_breakers.emplace_back(value);
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-target"}, "N",
        string_format("adaptive-p: select tokens near this probability (valid range 0.0 "
                      "to 1.0; negative = disabled) (default: %.2f)\n"
                      "[(more info)](https://github.com/ggml-org/llama.cpp/pull/17927)",
                      (double)params.sampling.adaptive_target),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_target = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-decay"}, "N",
        string_format("adaptive-p: decay rate for target adaptation over time. lower values "
                      "are more reactive, higher values are more stable.\n"
                      "(valid range 0.0 to 0.99) (default: %.2f)",
                      (double)params.sampling.adaptive_decay),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_decay = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dynatemp-range"}, "N",
        string_format("dynamic temperature range (default: %.2f, 0.0 = disabled)", (double)params.sampling.dynatemp_range),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_range = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dynatemp-exp"}, "N",
        string_format("dynamic temperature exponent (default: %.2f)", (double)params.sampling.dynatemp_exponent),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_exponent = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat"}, "N",
        string_format("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sampling.mirostat),
        [](common_params & params, int value) {
            params.sampling.mirostat = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-lr"}, "N",
        string_format("Mirostat learning rate, parameter eta (default: %.2f)", (double)params.sampling.mirostat_eta),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_eta = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-ent"}, "N",
        string_format("Mirostat target entropy, parameter tau (default: %.2f)", (double)params.sampling.mirostat_tau),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_tau = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sampling.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar"}, "GRAMMAR",
        "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, value};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar-file"}, "FNAME",
        "file to read grammar from",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, read_file(value)};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(value))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-jf", "--json-schema-file"}, "FILE",
        "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string schema;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(schema)
            );
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(schema))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_ROPE_SCALING_TYPE"));
    add_opt(common_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_SCALE"));
    add_opt(common_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](common_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_BASE"));
    add_opt(common_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_SCALE"));
    add_opt(common_arg(
        {"--yarn-orig-ctx"}, "N",
        string_format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](common_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ).set_env("LLAMA_ARG_YARN_ORIG_CTX"));
    add_opt(common_arg(
        {"--yarn-ext-factor"}, "N",
        string_format("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_EXT_FACTOR"));
    add_opt(common_arg(
        {"--yarn-attn-factor"}, "N",
        string_format("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double)params.yarn_attn_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_ATTN_FACTOR"));
    add_opt(common_arg(
        {"--yarn-beta-slow"}, "N",
        string_format("YaRN: high correction dim or alpha (default: %.2f)", (double)params.yarn_beta_slow),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_SLOW"));
    add_opt(common_arg(
        {"--yarn-beta-fast"}, "N",
        string_format("YaRN: low correction dim or beta (default: %.2f)", (double)params.yarn_beta_fast),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_FAST"));
    add_opt(common_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        string_format(
            "KV cache data type for K\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        string_format(
            "KV cache data type for V\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"-lm", "--load-mode"}, "MODE",
        "model loading mode (default: auto)\n"
        "- auto: mmap, unless a device does not support it\n"
        "- none: no special loading mode\n"
        "- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n"
        "- mlock: force system to keep model in RAM rather than swapping or compressing\n"
        "- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n"
        "- dio: use DirectIO if available\n",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "auto")       { params.load_mode = LLAMA_LOAD_MODE_AUTO;       }
            else if (value == "none")       { params.load_mode = LLAMA_LOAD_MODE_NONE;       }
            else if (value == "mmap")       { params.load_mode = LLAMA_LOAD_MODE_MMAP;       }
            else if (value == "mlock")      { params.load_mode = LLAMA_LOAD_MODE_MLOCK;      }
            else if (value == "mmap+mlock") { params.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK; }
            else if (value == "dio")        { params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;  }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_LOAD_MODE"));
    add_opt(common_arg(
        {"-lzm", "--lazy-mode"}, "MODE",
        "on-demand reading of certain tensors, for example per-layer embeddings (default: auto)\n"
        "- on: read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap)\n"
        "- auto: on, but only for tensors larger than 4 GiB\n"
        "- off: always keep them resident",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "on")   { params.lazy_mode = LLAMA_LAZY_MODE_ON;   }
            else if (value == "auto") { params.lazy_mode = LLAMA_LAZY_MODE_AUTO; }
            else if (value == "off")  { params.lazy_mode = LLAMA_LAZY_MODE_OFF;  }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_LAZY_MODE"));
    add_opt(common_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggml-org/llama.cpp/issues/1437",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_NUMA"));
    add_opt(common_arg(
        {"--list-devices"},
        "print list of available devices and exit",
        [](common_params &) {
            common_print_available_devices();
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--check-tensors"},
        string_format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](common_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(common_arg(
        {"--override-kv"}, "KEY=TYPE:VALUE,...",
        "advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                if (!string_parse_kv_override(item.c_str(), params.kv_overrides)) {
                    throw std::runtime_error(string_format("error: Invalid type for KV override: %s\n", item.c_str()));
                }
            }
        }
    ));
    add_opt(common_arg(
        {"-m", "--model"}, "FNAME",
        "model path to load",
        [](common_params & params, const std::string & value) {
            params.model.path = value;
        }
    ).set_env("LLAMA_ARG_MODEL"));
    add_opt(common_arg(
        {"--ple", "--shard"}, "FNAME,...",
        "extra GGUF shards for the model, comma separated; may be given more than once\n"
        "paths may live anywhere, including on different drives; --ple is an alias for --shard",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.model.shards.push_back(item);
            }
        }
    ).set_env("LLAMA_ARG_SHARD"));
    add_opt(common_arg(
        {"-o", "--output", "--output-file"}, "FNAME",
        "write generated text to a local file",
        [](common_params & params, const std::string & value) {
            params.out_file = value;
        }
    ));
    add_opt(common_arg(
        {"--chat-template-kwargs"}, "STRING",
        "sets additional params for the json template parser, must be a valid json object string, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'",
        [](common_params & params, const std::string & value) {
            auto parsed = json::parse(value);
            for (const auto & item : parsed.items()) {
                if (item.key() == "enable_thinking") {
                    LOG_WRN("Setting 'enable_thinking' via --chat-template-kwargs is deprecated. "
                            "Use --reasoning on / --reasoning off instead.\n");
                }
                if (item.key() == "preserve_reasoning") {
                    LOG_WRN("Setting 'preserve_reasoning' via --chat-template-kwargs is deprecated. "
                            "Use --reasoning-preserve / --no-reasoning-preserve instead.\n");
                }
                params.default_template_kwargs[item.key()] = item.value().dump();
            }
        }
    ).set_env("LLAMA_ARG_CHAT_TEMPLATE_KWARGS"));
    add_opt(common_arg(
        {"--jinja"},
        {"--no-jinja"},
        string_format("whether to use jinja template engine for chat (default: %s)", params.use_jinja ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_jinja = value;
        }
    ).set_env("LLAMA_ARG_JINJA"));
    add_opt(common_arg(
        {"-rea", "--reasoning"}, "[on|off|auto]",
        "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.enable_reasoning = 1;
                params.default_template_kwargs["enable_thinking"] = "true";
            } else if (is_falsey(value)) {
                params.enable_reasoning = 0;
                params.default_template_kwargs["enable_thinking"] = "false";
            } else if (is_autoy(value)) {
                params.enable_reasoning = -1;
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --reasoning: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_REASONING"));
    add_opt(common_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        string_format(
            "set custom jinja chat template (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = value;
        }
    ).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(common_arg(
        {"--chat-template-file"}, "JINJA_TEMPLATE_FILE",
        string_format(
            "set custom jinja chat template file (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = read_file(value);
        }
    ).set_env("LLAMA_ARG_CHAT_TEMPLATE_FILE"));
    add_opt(common_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](common_params & params) {
            params.simple_io = true;
        }
    ));
    add_opt(common_arg(
        {"--log-disable"},
        "Log disable",
        [](common_params &) {
            common_log_pause(common_log_main());
        }
    ));
    add_opt(common_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](common_params &, const std::string & value) {
            common_log_set_file(common_log_main(), value.c_str());
        }
    ).set_env("LLAMA_ARG_LOG_FILE"));
    add_opt(common_arg(
        {"--log-colors"}, "[on|off|auto]",
        "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params &, const std::string & value) {
            if (is_truthy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_ENABLED);
            } else if (is_falsey(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_DISABLED);
            } else if (is_autoy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_AUTO);
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --log-colors: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_LOG_COLORS"));
    add_opt(common_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](common_params & params) {
            params.verbosity = INT_MAX;
            common_log_set_verbosity_thold(INT_MAX);
        }
    ));
    add_opt(common_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        string_format("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n"
            " - 0: generic output\n"
            " - 1: error\n"
            " - 2: warning\n"
            " - 3: info\n"
            " - 4: trace (more info)\n"
            " - 5: debug\n"
            "(default: %d)\n", params.verbosity),
        [](common_params & params, int value) {
            params.verbosity = value;
            common_log_set_verbosity_thold(value);
        }
    ).set_env("LLAMA_ARG_LOG_VERBOSITY"));
    add_opt(common_arg(
        {"--log-prefix"},
        {"--no-log-prefix"},
        "Enable prefix in log messages",
        [](common_params &, bool value) {
            common_log_set_prefix(common_log_main(), value);
        }
    ).set_env("LLAMA_ARG_LOG_PREFIX"));
    add_opt(common_arg(
        {"--log-timestamps"},
        {"--no-log-timestamps"},
        "Enable timestamps in log messages",
        [](common_params &, bool value) {
            common_log_set_timestamps(common_log_main(), value);
        }
    ).set_env("LLAMA_ARG_LOG_TIMESTAMPS"));

    return ctx_arg;
}
