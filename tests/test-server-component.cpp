#include "testing.h"

#include "../tools/server/server-chat.h"
#include "../tools/server/server-common.h"
#include "../tools/server/server-decision.h"
#include "../tools/server/server-schema.h"
#include "../tools/server/server-responses.h"
#include "../tools/server/server-chat-completions.h"
#include "../tools/server/server-conversations.h"
#include "../tools/server/server-files.h"
#include "../tools/server/server-files-http.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// unit tests for server internal components, no model required
// NOTE: end-to-end tests live in tools/server/tests
// see skills/code-review/SKILL.md for more

//
// server_tokens (text only, has_mtmd = false)
//

static void test_server_tokens_basic(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    t.assert_equal("size", (size_t) 3, st.size());
    t.assert_true("not empty", !st.empty());
    t.assert_equal("operator[]", 2, st[1]);

    st.push_back(4);
    t.assert_equal("push_back", (size_t) 4, st.size());
    t.assert_true("get_tokens", st.get_tokens() == llama_tokens({1, 2, 3, 4}));

    st.insert({5, 6});
    t.assert_true("insert appends", st.get_tokens() == llama_tokens({1, 2, 3, 4, 5, 6}));

    st.set_token(0, 9);
    t.assert_equal("set_token", 9, st[0]);

    bool thrown = false;
    try {
        st.push_back(LLAMA_TOKEN_NULL);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("push_back(LLAMA_TOKEN_NULL) throws", thrown);
    t.assert_equal("size unchanged after throw", (size_t) 6, st.size());

    st.clear();
    t.assert_true("clear", st.empty());
}

static void test_server_tokens_pos(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    t.assert_equal("pos_next()", (llama_pos) 5, st.pos_next());
    t.assert_equal("pos_next(0)", (llama_pos) 0, st.pos_next(0));
    t.assert_equal("pos_next(3)", (llama_pos) 3, st.pos_next(3));

    t.assert_equal("size_up_to_pos(0)", (size_t) 0, st.size_up_to_pos(0));
    t.assert_equal("size_up_to_pos(3)", (size_t) 3, st.size_up_to_pos(3));
    t.assert_equal("size_up_to_pos clamps", (size_t) 5, st.size_up_to_pos(100));
}

static void test_server_tokens_keep_first(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    st.keep_first(5);
    t.assert_equal("keep all", (size_t) 5, st.size());
    st.keep_first(2);
    t.assert_true("keep 2", st.get_tokens() == llama_tokens({1, 2}));
    st.keep_first(0);
    t.assert_true("keep 0", st.empty());
}

static void test_server_tokens_common_prefix(testing & t) {
    server_tokens a(llama_tokens{1, 2, 3, 4}, false);
    server_tokens empty(llama_tokens{}, false);

    t.assert_equal("both empty",  (size_t) 0, empty.get_common_prefix(empty));
    t.assert_equal("one empty",   (size_t) 0, a.get_common_prefix(empty));
    t.assert_equal("identical",   (size_t) 4, a.get_common_prefix(a));
    t.assert_equal("diverge at 2", (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 9, 4}, false)));
    t.assert_equal("diverge at 0", (size_t) 0, a.get_common_prefix(server_tokens(llama_tokens{9, 2, 3, 4}, false)));
    t.assert_equal("b is prefix",  (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2}, false)));
    t.assert_equal("a is prefix",  (size_t) 4, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 3, 4, 5, 6}, false)));
}

static void test_server_tokens_serialize(testing & t) {
    // same packing as slot save/restore in server-context.cpp
    auto pack = [](const std::vector<char> & data) {
        llama_tokens packed(data.size() / sizeof(llama_token));
        memcpy(packed.data(), data.data(), data.size());
        return packed;
    };

    t.test("roundtrip", [&](testing & t) {
        server_tokens st(llama_tokens{10, 20, 30}, false);
        std::vector<char> data = st.serialize();
        t.assert_equal("aligned to llama_token", (size_t) 0, data.size() % sizeof(llama_token));

        server_tokens restored = server_tokens::deserialize(pack(data), false);
        t.assert_true("tokens", restored.get_tokens() == st.get_tokens());
    });

    t.test("roundtrip empty", [&](testing & t) {
        server_tokens st(llama_tokens{}, false);
        server_tokens restored = server_tokens::deserialize(pack(st.serialize()), false);
        t.assert_true("empty", restored.empty());
    });

    t.test("legacy plain token list", [&](testing & t) {
        server_tokens restored = server_tokens::deserialize(llama_tokens{7, 8, 9}, false);
        t.assert_true("tokens", restored.get_tokens() == llama_tokens({7, 8, 9}));
    });

    auto expect_throw = [](testing & t, const std::string & msg, const llama_tokens & packed) {
        bool thrown = false;
        try {
            server_tokens::deserialize(packed, false);
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true(msg, thrown);
    };

    t.test("corrupted", [&](testing & t) {
        llama_tokens good = pack(server_tokens(llama_tokens{10, 20, 30}, false).serialize());

        llama_tokens bad_version = good;
        bad_version[1] = 999;
        expect_throw(t, "bad version", bad_version);

        llama_tokens truncated = good;
        truncated.pop_back();
        expect_throw(t, "truncated", truncated);

        llama_tokens trailing = good;
        trailing.push_back(42);
        expect_throw(t, "trailing data", trailing);

        llama_tokens huge_count = good;
        huge_count[2] = 0x7fffffff;
        expect_throw(t, "huge token count", huge_count);

        expect_throw(t, "marker only", llama_tokens{LLAMA_TOKEN_NULL});
    });
}

static void test_server_tokens_clone(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    server_tokens cl = st.clone();
    cl.push_back(4);
    t.assert_equal("original untouched", (size_t) 3, st.size());
    t.assert_equal("clone grew", (size_t) 4, cl.size());
    t.assert_true("get_text_tokens", cl.get_text_tokens() == llama_tokens({1, 2, 3, 4}));
}

static void test_server_tokens(testing & t) {
    t.test("basic",         test_server_tokens_basic);
    t.test("pos",           test_server_tokens_pos);
    t.test("keep_first",    test_server_tokens_keep_first);
    t.test("common_prefix", test_server_tokens_common_prefix);
    t.test("serialize",     test_server_tokens_serialize);
    t.test("clone",         test_server_tokens_clone);
}

//
// string / json helpers
//

static void test_utf8(testing & t) {
    t.test("validate_utf8", [](testing & t) {
        t.assert_equal("empty",            (size_t) 0, validate_utf8(""));
        t.assert_equal("ascii",            (size_t) 3, validate_utf8("abc"));
        t.assert_equal("full 2-byte",      (size_t) 3, validate_utf8("a\xC3\xA9"));
        t.assert_equal("full 3-byte",      (size_t) 4, validate_utf8("a\xE2\x82\xAC"));
        t.assert_equal("full 4-byte",      (size_t) 5, validate_utf8("a\xF0\x9F\x98\x80"));
        t.assert_equal("cut 2-byte (1/2)", (size_t) 1, validate_utf8("a\xC3"));
        t.assert_equal("cut 3-byte (1/3)", (size_t) 1, validate_utf8("a\xE2"));
        t.assert_equal("cut 3-byte (2/3)", (size_t) 1, validate_utf8("a\xE2\x82"));
        t.assert_equal("cut 4-byte (1/4)", (size_t) 1, validate_utf8("a\xF0"));
        t.assert_equal("cut 4-byte (2/4)", (size_t) 1, validate_utf8("a\xF0\x9F"));
        t.assert_equal("cut 4-byte (3/4)", (size_t) 1, validate_utf8("a\xF0\x9F\x98"));
        t.assert_equal("cut only",         (size_t) 0, validate_utf8("\xF0\x9F\x98"));
        t.assert_equal("full then cut",    (size_t) 4, validate_utf8("\xF0\x9F\x98\x80\xC3"));
    });

    t.test("is_valid_utf8", [](testing & t) {
        t.assert_true("empty",          is_valid_utf8(""));
        t.assert_true("ascii",          is_valid_utf8("hello"));
        t.assert_true("multi-byte",     is_valid_utf8("\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80"));
        t.assert_true("truncated",      !is_valid_utf8("\xE2\x82"));
        t.assert_true("bad cont. byte", !is_valid_utf8("\xC3\x41"));
        t.assert_true("lone cont. byte", !is_valid_utf8("\x80"));
        t.assert_true("invalid lead",   !is_valid_utf8("\xFF"));
    });
}

static void test_sse(testing & t) {
    t.test("oai", [](testing & t) {
        t.assert_equal("single", std::string("data: {\"a\":1}\n\n"), format_oai_sse(json{{"a", 1}}));
        t.assert_equal("array",  std::string("data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"),
                format_oai_sse(json::array({json{{"a", 1}}, json{{"b", 2}}})));
        t.assert_equal("invalid utf8 replaced", std::string("data: \"x\xEF\xBF\xBD\"\n\n"), format_oai_sse(json("x\xC3")));
    });

    t.test("oai_resp", [](testing & t) {
        json ev = {{"event", "response.created"}, {"data", {{"id", "r1"}}}};
        t.assert_equal("single", std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\n"), format_oai_resp_sse(ev));
        t.assert_equal("array",  std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\nevent: response.created\ndata: {\"id\":\"r1\"}\n\n"),
                format_oai_resp_sse(json::array({ev, ev})));
    });

    t.test("anthropic", [](testing & t) {
        json ev = {{"event", "message_start"}, {"data", {{"type", "message_start"}}}};
        t.assert_equal("with event", std::string("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"), format_anthropic_sse(ev));
        t.assert_equal("data only",  std::string("data: {\"type\":\"ping\"}\n\n"), format_anthropic_sse(json{{"type", "ping"}}));
    });
}

static void test_json_helpers(testing & t) {
    t.test("json_is_array_of_numbers", [](testing & t) {
        t.assert_true("ints",         json_is_array_of_numbers(json::array({1, 2, 3})));
        t.assert_true("empty array",  json_is_array_of_numbers(json::array()));
        t.assert_true("float",        !json_is_array_of_numbers(json::array({1, 2.5})));
        t.assert_true("string",       !json_is_array_of_numbers(json::array({1, "a"})));
        t.assert_true("not an array", !json_is_array_of_numbers(json(1)));
    });

    t.test("json_is_array_of_mixed_numbers_strings", [](testing & t) {
        t.assert_true("mixed",        json_is_array_of_mixed_numbers_strings(json::array({"a", 1})));
        t.assert_true("numbers only", !json_is_array_of_mixed_numbers_strings(json::array({1, 2})));
        t.assert_true("strings only", !json_is_array_of_mixed_numbers_strings(json::array({"a", "b"})));
        t.assert_true("empty",        !json_is_array_of_mixed_numbers_strings(json::array()));
    });

    t.test("json_is_array_and_contains_numbers", [](testing & t) {
        t.assert_true("contains",     json_is_array_and_contains_numbers(json::array({"a", 1})));
        t.assert_true("strings only", !json_is_array_and_contains_numbers(json::array({"a"})));
        t.assert_true("not an array", !json_is_array_and_contains_numbers(json(1)));
    });

    t.test("json_get_nested_values", [](testing & t) {
        json js = {{"a", {{"b", {{"c", 1}}}}}, {"x", 2}};
        json res = json_get_nested_values({"a/b/c", "x", "a/b", "a/missing", "x/y"}, js);
        t.assert_equal("nested",  1, res.at("a/b/c").get<int>());
        t.assert_equal("top",     2, res.at("x").get<int>());
        t.assert_true("object",   res.at("a/b") == json{{"c", 1}});
        t.assert_true("missing",  !res.contains("a/missing"));
        t.assert_true("not obj",  !res.contains("x/y"));
        t.assert_equal("count", (size_t) 3, res.size());
    });
}

static void test_lora_helpers(testing & t) {
    // only pointer identity is used, never dereferenced
    auto * p0 = reinterpret_cast<llama_adapter_lora *>(0x10);
    auto * p1 = reinterpret_cast<llama_adapter_lora *>(0x20);

    auto make = [](llama_adapter_lora * ptr, float scale) {
        common_adapter_lora_info info;
        info.scale = scale;
        info.ptr   = ptr;
        return info;
    };

    t.test("parse_lora_request", [](testing & t) {
        auto res = parse_lora_request(json::array({
            json{{"id", 0}, {"scale", 0.5}},
            json{{"id", 2}, {"scale", 1.0}},
            json{{"id", 0}, {"scale", 0.25}}, // last one wins
        }));
        t.assert_equal("count", (size_t) 2, res.size());
        t.assert_equal("id 0", 0.25f, res.at(0));
        t.assert_equal("id 2", 1.0f,  res.at(2));

        auto defaults = parse_lora_request(json::array({json::object()}));
        t.assert_equal("default id/scale", 0.0f, defaults.at(-1));
    });

    t.test("are_lora_equal", [&](testing & t) {
        t.assert_true("both empty",   are_lora_equal({}, {}));
        t.assert_true("same",         are_lora_equal({make(p0, 1.0f)}, {make(p0, 1.0f)}));
        t.assert_true("diff scale",   !are_lora_equal({make(p0, 1.0f)}, {make(p0, 0.5f)}));
        t.assert_true("diff ptr",     !are_lora_equal({make(p0, 1.0f)}, {make(p1, 1.0f)}));
        t.assert_true("diff size",    !are_lora_equal({make(p0, 1.0f)}, {}));
    });

    t.test("lora_get_enabled_ids", [&](testing & t) {
        auto ids = lora_get_enabled_ids({make(p0, 0.0f), make(p1, 0.5f), make(p0, -1.0f), make(p1, 2.0f)});
        t.assert_true("ids", ids == std::vector<size_t>({1, 3}));
    });
}

//
// server_pipe
//

static void test_server_pipe(testing & t) {
    auto stop_now = []() { return true; };

    t.test("drain then eof", [](testing & t) {
        server_pipe<int> p;
        t.assert_true("write 1", p.write(1));
        t.assert_true("write 2", p.write(2));
        p.close_write();

        int v = 0;
        t.assert_true("read 1", p.read(v, nullptr) && v == 1);
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("broken pipe", [](testing & t) {
        server_pipe<int> p;
        p.close_read();
        t.assert_true("write fails", !p.write(1));
    });

    t.test("max_size drops oldest", [](testing & t) {
        server_pipe<int> p;
        p.max_size = 2;
        p.write(1);
        p.write(2);
        p.write(3);
        p.close_write();

        int v = 0;
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("read 3", p.read(v, nullptr) && v == 3);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("queued data wins over should_stop", [&](testing & t) {
        server_pipe<int> p;
        p.write(1);
        int v = 0;
        t.assert_true("read", p.read(v, stop_now) && v == 1);
    });

    t.test("should_stop closes the pipe", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now));
        t.assert_true("write fails", !p.write(1));
    });

    t.test("should_stop as deadline keeps the pipe open", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now, /* close_on_stop */ false));
        t.assert_true("write ok", p.write(1));
        t.assert_true("read again", p.read(v, nullptr) && v == 1);
    });

    t.test("cross thread", [](testing & t) {
        constexpr int n = 1000;
        server_pipe<int> p;
        std::thread producer([&]() {
            for (int i = 0; i < n; i++) {
                p.write(int(i));
            }
            p.close_write();
        });

        int v = 0;
        int count = 0;
        bool in_order = true;
        while (p.read(v, nullptr)) {
            in_order &= v == count;
            count++;
        }
        producer.join();

        t.assert_equal("count", n, count);
        t.assert_true("in order", in_order);
    });
}

//
// server_decision_escape_special_tokens
//

// "<|name|>" with U+00A6 in place of "|"
static std::string escaped(const std::string & name) {
    return "<\xC2\xA6" + name + "\xC2\xA6>";
}

static void test_decision_escape(testing & t) {
    t.test("cases", [](testing & t) {
        const std::vector<std::pair<std::string, std::string>> cases = {
            { "",                 ""                               },
            { "no tokens",        "no tokens"                      },
            { "<|im_start|>user", escaped("im_start") + "user"     },
            { "<|a_1|><|B|>",     escaped("a_1") + escaped("B")    },
            { "<|<|a|>",          "<|" + escaped("a")              },
            { "<|a|>|>",          escaped("a") + "|>"              },
            { "<||>",             "<||>"                           },
            { "<|a b|>",          "<|a b|>"                        },
            { "<|a|b|>",          "<|a|b|>"                        },
            { "<|caf\xC3\xA9|>",  "<|caf\xC3\xA9|>"                },
            { "<|abc",            "<|abc"                          },
        };
        for (const auto & [text, expected] : cases) {
            t.assert_equal(text, expected, server_decision_escape_special_tokens(text));
        }
    });

    // every string of up to 6 chars from "<|>a-" gives the text of the regex
    t.test("same as std::regex", [](testing & t) {
        static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
        const std::string chars = "<|>a-";
        size_t n_strings = 0;
        size_t n_diff    = 0;
        for (size_t len = 0; len <= 6; len++) {
            size_t n = 1;
            for (size_t k = 0; k < len; k++) {
                n *= chars.size();
            }
            for (size_t idx = 0; idx < n; idx++) {
                std::string text;
                for (size_t k = 0, v = idx; k < len; k++, v /= chars.size()) {
                    text += chars[v % chars.size()];
                }
                if (server_decision_escape_special_tokens(text) != std::regex_replace(text, re_special, "<\xC2\xA6$1\xC2\xA6>")) {
                    n_diff++;
                }
                n_strings++;
            }
        }
        t.assert_equal("strings that differ, of " + std::to_string(n_strings), (size_t) 0, n_diff);
    });

    // std::regex overflowed the stack on these
    t.test("long input", [](testing & t) {
        const std::string name(1000000, 'a');
        t.assert_true("not closed", server_decision_escape_special_tokens("<|" + name) == "<|" + name);
        t.assert_true("closed", server_decision_escape_special_tokens("<|" + name + "|>") == escaped(name));
    });
}

static void test_rerank_template(testing & t) {
    struct test_case {
        const char * name;
        std::string tmpl;
        std::string query;
        std::string doc;
        std::string expected;
    };
    const test_case cases[] = {
        {"plain inputs", "Q: {query}\nD: {document}", "query", "doc", "Q: query\nD: doc"},
        {"document placeholder in query", "Q: {query}\nD: {document}", "Explain {document}.", "doc", "Q: Explain {document}.\nD: doc"},
        {"query placeholder in query", "{query}|{document}", "{query}", "doc", "{query}|doc"},
        {"placeholders in both inputs", "{query}|{document}", "{query} {document}", "{query} {document}", "{query} {document}|{query} {document}"},
        {"repeated placeholders", "{query}|{document}|{query}|{document}", "{document}", "doc", "{document}|doc|{document}|doc"},
        {"adjacent reversed placeholders", "{document}{query}", "query", "doc", "docquery"},
        {"query only", "{query}", "{document}", "doc", "{document}"},
        {"document only", "{document}", "query", "{query}", "{query}"},
        {"no placeholders", "fixed", "{document}", "{query}", "fixed"},
        {"unknown and incomplete placeholders", "{unknown}|{query}|{document}|{query", "query", "doc", "{unknown}|query|doc|{query"},
        {"nested braces", "{{query}}|{{document}}", "query", "doc", "{query}|{doc}"},
        {"placeholder formed at boundary", "{query}ument}", "{doc", "doc", "{document}"},
        {"empty inputs", "Q: {query}\nD: {document}", "", "", "Q: \nD: "},
        {"empty template", "", "query", "doc", ""},
    };
    for (const auto & test : cases) {
        t.assert_equal(test.name, test.expected, format_rerank_template(test.tmpl, test.query, test.doc));
    }
}

//
// server_decision_context::format_answer, 2 class scores (minicpm-ocr-jev)
//

static void test_decision_class_scores(testing & t) {
    server_decision_context ctx;
    ctx.type = COMMON_DECISION_TYPE_MINICPM_OCR_JEV;

    const server_decision_question choice = {"q", SERVER_DECISION_QUESTION_CHOICE, json(), {{"0", json()}, {"1", json()}}};
    const server_decision_question noul   = {"q", SERVER_DECISION_QUESTION_NOUL,   json(), {{"false", json()}, {"true", json()}}};

    const std::vector<std::pair<float, float>> cases = {
        {0.0f, 0.0f}, {-1.5f, 2.25f}, {3.0f, -0.5f}, {80.0f, -80.0f}, {1000.0f, 1001.0f},
    };
    for (const auto & [z0, z1] : cases) {
        const double p1  = 1.0 / (1.0 + std::exp((double) z0 - (double) z1));
        const json   a   = ctx.format_answer(choice, {{z0, z1}});
        const double ap0 = a.at("probabilities").at("0").get<double>();
        const double ap1 = a.at("probabilities").at("1").get<double>();
        t.assert_true("p1", std::fabs(ap1 - p1) <= 1e-6);
        t.assert_true("sum", std::fabs(ap0 + ap1 - 1.0) <= 1e-9);
        // a tie picks the first class
        t.assert_equal("choice", std::string(z1 > z0 ? "1" : "0"), a.at("choice").get<std::string>());
        t.assert_true("confidence", std::fabs(a.at("confidence").get<double>() - (2.0 * std::max(ap0, ap1) - 1.0)) <= 1e-9);

        const json n = ctx.format_answer(noul, {{z0, z1}});
        t.assert_true("noul is p1", n.at("noul").get<double>() == ap1);
    }

    for (const float bad : {std::nanf(""), INFINITY}) {
        bool thrown = false;
        try {
            ctx.format_answer(choice, {{bad, 0.0f}});
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true("nan or +inf", thrown);
    }
}

//
// server_decision_context::parse_questions / parse_state, the request checks of minicpm-ocr-jev
//

static void test_decision_ocr_jev_validation(testing & t) {
    server_decision_context ctx;
    ctx.type = COMMON_DECISION_TYPE_MINICPM_OCR_JEV;

    // only base64-decoded, the image itself is not parsed here
    const std::string image = "data:image/png;base64,AAAA";

    const json choice = {
        {"type",     "choice"},
        {"criteria", json{{"0", nullptr}, {"1", nullptr}}},
    };
    const auto questions_of = [](const json & question) {
        return json{{"state", "some OCR text"}, {"questions", json{{"q", question}}}};
    };
    const auto files_of = [&](size_t n) {
        json files = json::array();
        for (size_t i = 0; i < n; i++) {
            files.push_back(image);
        }
        return json{{"state", "some OCR text"}, {"files", files}};
    };

    const auto reject_questions = [&](const char * name, const json & body, const std::string & expected) {
        std::string error;
        try {
            ctx.parse_questions(body);
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        t.assert_true(std::string(name) + ": " + error, error.find(expected) != std::string::npos);
    };
    const auto reject_state = [&](const char * name, const json & body, const std::string & expected) {
        std::vector<raw_buffer> files;
        std::string error;
        try {
            ctx.parse_state(body, files);
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        t.assert_true(std::string(name) + ": " + error, error.find(expected) != std::string::npos);
    };

    reject_state("state is text", {{"state", 42}}, "\"state\" must be the OCR text");

    // one fixed question, no instructions
    reject_questions("one question", {{"state", "x"}, {"questions", json{{"a", choice}, {"b", choice}}}},
        "answers exactly one question");
    reject_questions("no instructions",
        questions_of(json{{"type", "choice"}, {"instructions", "judge it"}, {"criteria", json{{"0", nullptr}, {"1", nullptr}}}}),
        "\"instructions\" is not used by this model");

    // the classes are fixed, so are the criteria
    reject_questions("criteria are the classes",
        questions_of(json{{"type", "choice"}, {"criteria", json{{"yes", nullptr}, {"no", nullptr}}}}),
        "\"criteria\" must be {\"0\": null, \"1\": null}");
    reject_questions("noul without criteria",
        questions_of(json{{"type", "noul"}, {"criteria", json{{"false", "no"}, {"true", "yes"}}}}),
        "\"criteria\" is not supported for a noul question");
    reject_questions("choice or noul only",
        questions_of(json{{"type", "score"}, {"criteria", json::array({"low", "high"})}}),
        "only choice and noul questions are supported");

    // the prompt has the original image, then the one rendered from the OCR text
    reject_state("two images", files_of(1), "this model needs exactly 2 images");
    reject_state("two images, three given", files_of(3), "this model needs exactly 2 images");
}

// server_decision_context: the recursive functions stop at RECURSIVE_LIMIT
//

static void test_decision_depth(testing & t) {
    const auto prepare = [](size_t depth) {
        server_decision_context ctx;
        ctx.type = COMMON_DECISION_TYPE_H2O_LIGHTNING; // the state goes through decision_py_dump
        std::string error;
        try {
            ctx.prepare_state(json::parse(std::string(depth, '[') + std::string(depth, ']')));
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        return error;
    };
    t.test("state at the limit", [&](testing & t) {
        const std::string error = prepare(128);
        t.assert_true("accepted: " + error, error.empty());
    });
    t.test("deep state", [&](testing & t) {
        const std::string error = prepare(1000);
        t.assert_true("rejected as nested: " + error, error.find("nested more than 128 levels deep") != std::string::npos);
    });
}

// DRY sequence breakers
//

static void test_dry_sequence_breakers(testing & t) {
    common_params params_base;

    const auto eval = [&](const json & data) {
        return server_schema::eval_llama_cmpl_schema(nullptr, params_base, {}, data);
    };

    {
        json data;
        data["dry_sequence_breakers"] = json::array({"ab", "ab", "cd", "ab"});
        const auto params = eval(data);
        const std::vector<std::string> expected = {"ab", "cd"};
        t.assert_true("duplicates removed, order kept", params.sampling.dry_sequence_breakers == expected);
    }

    {
        std::vector<std::string> many;
        for (int i = 0; i <= LLAMA_DRY_MAX_SEQ_BREAKERS; ++i) {
            many.push_back("b" + std::to_string(i));
        }
        json data;
        data["dry_sequence_breakers"] = many;

        bool thrown = false;
        try {
            eval(data);
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true("more than the cap is rejected", thrown);
    }
}

//
// glob_match
//

static void test_glob_match(testing & t) {
    t.test("patterns", [](testing & t) {
        t.assert_true("literal", glob_match("src/main.cpp", "src/main.cpp"));
        t.assert_true("star", glob_match("src/*.cpp", "src/main.cpp"));
        t.assert_true("star stops at /", !glob_match("*.cpp", "src/main.cpp"));
        t.assert_true("double star", glob_match("**/*.cpp", "a/b/c/main.cpp"));
        t.assert_true("double star at end", glob_match("src/**", "src/a/b"));
        t.assert_true("question mark", glob_match("ma?n.cpp", "main.cpp"));
        t.assert_true("question mark not /", !glob_match("a?b", "a/b"));
        t.assert_true("class", glob_match("[a-c]x", "bx"));
        t.assert_true("negated class", !glob_match("[!a-c]x", "bx"));
        t.assert_true("unclosed class", glob_match("[ab", "[ab"));
        t.assert_true("mismatch", !glob_match("main.cpp", "main.c"));
    });
    t.test("long input", [](testing & t) {
        const std::string path(100000, 'a');
        t.assert_true("long literal", glob_match(path, path));
        t.assert_true("long path", glob_match("**", path));
        t.assert_true("too many wildcards", !glob_match(std::string(1000, '*'), "a"));
    });
}

//
// OpenAI speech API
//

static void test_speech_convert(testing & t) {
    const json base = {{"model", "tts"}, {"input", "Hello"}, {"voice", "alba"}};

    auto with = [&](const char * key, const json & value) {
        json body = base;
        body[key] = value;
        return body;
    };
    auto rejected = [&](const char * key, const json & value) {
        try {
            convert_speech_to_tts(with(key, value));
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };

    // defaults
    json out = convert_speech_to_tts(base);
    t.assert_equal("input kept",        std::string("Hello"), out.at("input").get<std::string>());
    t.assert_equal("voice kept",        std::string("alba"),  out.at("voice").get<std::string>());
    t.assert_equal("default format",    std::string("wav"),   out.at("response_format").get<std::string>());
    t.assert_true ("no stream",         !out.contains("stream"));

    // response_format
    out = convert_speech_to_tts(with("response_format", "wav"));
    t.assert_equal("wav",               std::string("wav"),   out.at("response_format").get<std::string>());
    out = convert_speech_to_tts(with("response_format", "pcm"));
    t.assert_equal("pcm is 16-bit",     std::string("pcm16"), out.at("response_format").get<std::string>());
    for (const char * format : {"mp3", "opus", "aac", "flac", "pcm16"}) {
        t.assert_true(std::string("rejects ") + format, rejected("response_format", format));
    }

    // custom voice
    out = convert_speech_to_tts(with("voice", json{{"id", "voice_1234"}}));
    t.assert_equal("voice object",      std::string("voice_1234"), out.at("voice").get<std::string>());

    // stream_format
    out = convert_speech_to_tts(with("stream_format", "audio"));
    t.assert_true ("audio streams",     out.at("stream").get<bool>());
    t.assert_true ("rejects sse",       rejected("stream_format", "sse"));

    // speed
    t.assert_true ("speed 1.0",         !rejected("speed", 1.0));
    t.assert_true ("rejects speed 1.5", rejected("speed", 1.5));

    // instructions are ignored
    t.assert_true ("instructions",      !rejected("instructions", "whisper"));
}

static void test_pcm_f32_to_s16(testing & t) {
    auto sample = [](const std::string & bytes, size_t i) {
        int16_t v;
        memcpy(&v, bytes.data() + i * sizeof(v), sizeof(v));
        return (int) v;
    };

    const std::vector<float> samples = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -2.0f};
    const std::string bytes = pcm_f32_to_s16(samples.data(), samples.size());
    t.assert_equal("size",           (size_t) 12, bytes.size());
    t.assert_equal("zero",           0,      sample(bytes, 0));
    t.assert_equal("max",            32767,  sample(bytes, 1));
    t.assert_equal("min",            -32767, sample(bytes, 2));
    t.assert_equal("rounds",         16384,  sample(bytes, 3));
    t.assert_equal("clamps above",   32767,  sample(bytes, 4));
    t.assert_equal("clamps below",   -32767, sample(bytes, 5));
    t.assert_true ("empty",          pcm_f32_to_s16(nullptr, 0).empty());
}


//
// server_responses
//

static bool has_event(const json & events, const std::string & type) {
    for (const auto & event : events) {
        if (event.value("event", std::string()) == type) {
            return true;
        }
    }
    return false;
}

static void test_responses_parse(testing & t) {
    json body = {
        {"model", "test-model"},
        {"input", "hello"},
        {"max_output_tokens", 8},
        {"temperature", 0.5},
        {"metadata", json { {"user", "1"} }},
        {"tools", json::array({ json {
            {"type", "function"},
            {"name", "get_weather"},
            {"description", "weather"},
            {"parameters", json { {"type", "object"} }},
        } })},
        {"tool_choice", "auto"},
    };

    server_responses_request req = server_responses_parse_request(body);
    t.assert_equal("model",        std::string("test-model"), req.model);
    t.assert_equal("input items",  (size_t) 1, req.input_items.size());
    t.assert_equal("tool count",   (size_t) 1, req.tools_parsed.size());
    t.assert_equal("tool name",    std::string("get_weather"), req.tools_parsed[0].name);
    t.assert_equal("max tokens",   8, req.max_output_tokens);
    t.assert_true ("store default", req.store);
    t.assert_true ("input item id", req.input_items[0].contains("id"));
    t.assert_true ("stream default", !req.stream);

    bool thrown = false;
    json bad = body;
    bad["temperature"] = 3.0;
    try {
        server_responses_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("temperature bound", thrown);

    json background_body = body;
    background_body["background"] = true;
    server_responses_request background_req = server_responses_parse_request(background_body);
    t.assert_true("background accepted", background_req.background);

    bad = background_body;
    bad["stream"] = true;
    server_responses_request background_stream_req = server_responses_parse_request(bad);
    t.assert_true("background stream accepted", background_stream_req.background && background_stream_req.stream);

    json auto_trunc = body;
    auto_trunc["truncation"] = "auto";
    t.assert_equal("truncation auto", std::string("auto"), server_responses_parse_request(auto_trunc).truncation);

    thrown = false;
    bad = body;
    bad.erase("input");
    try {
        server_responses_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("input required", thrown);

    json named = body;
    named["tool_choice"] = json { {"type", "function"}, {"name", "get_weather"} };
    req = server_responses_parse_request(named);
    t.assert_equal("named choice", std::string("get_weather"), req.tool_choice_name);
}

static void test_responses_previous(testing & t) {
    server_responses_store store;
    server_responses_stored previous;
    previous.input_items = json::array({ json {
        {"type", "message"},
        {"role", "user"},
        {"content", json::array({ json { {"type", "input_text"}, {"text", "previous question"} } })},
    } });
    previous.response = json { {"output", json::array({ json {
        {"type", "message"},
        {"role", "assistant"},
        {"content", json::array({ json { {"type", "output_text"}, {"text", "previous answer"} } })},
    } }) } };
    store.put("resp_prev", previous);

    server_responses_request req = server_responses_parse_request(json {
        {"model", "m"},
        {"input", "next"},
        {"previous_response_id", "resp_prev"},
    });
    server_responses_resolve_input(req, store);
    t.assert_equal("effective items", (size_t) 3, req.effective_input.size());

    bool thrown = false;
    server_responses_request unknown = server_responses_parse_request(json {
        {"model", "m"},
        {"input", "next"},
        {"previous_response_id", "resp_missing"},
    });
    try {
        server_responses_resolve_input(unknown, store);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("unknown previous rejected", thrown);
}

static void test_responses_stream(testing & t) {
    server_responses_store store;
    server_responses_request req;
    req.model  = "m";
    req.stream = true;
    req.store  = false;

    server_responses_context ctx(store, req, "m", nullptr);

    common_chat_msg_diff reasoning_diff;
    reasoning_diff.reasoning_content_delta = "think";
    common_chat_msg_diff text_diff;
    text_diff.content_delta = "hello";
    common_chat_msg_diff tool_diff;
    tool_diff.tool_call_index      = 0;
    tool_diff.tool_call_delta.id   = "call_1";
    tool_diff.tool_call_delta.name = "f";
    tool_diff.tool_call_delta.arguments = "{\"x\":1}";

    server_task_result_cmpl_partial partial;
    partial.oaicompat_msg_diffs = { reasoning_diff, text_diff, tool_diff };

    json events = ctx.consume(partial);
    t.assert_true("created",        has_event(events, "response.created"));
    t.assert_true("in progress",    has_event(events, "response.in_progress"));
    t.assert_true("reasoning item", has_event(events, "response.output_item.added"));
    t.assert_true("reasoning text", has_event(events, "response.reasoning_text.delta"));
    t.assert_true("text delta",     has_event(events, "response.output_text.delta"));
    t.assert_true("tool delta",     has_event(events, "response.function_call_arguments.delta"));

    server_task_result_cmpl_final final_result;
    final_result.n_prompt_tokens = 4;
    final_result.n_decoded       = 7;
    final_result.stop            = STOP_TYPE_EOS;
    final_result.truncated       = false;
    final_result.oaicompat_msg.reasoning_content = "think";
    final_result.oaicompat_msg.content           = "hello";

    json done = ctx.finish(final_result);
    t.assert_true("text done",     has_event(done, "response.output_text.done"));
    t.assert_true("tool done",     has_event(done, "response.function_call_arguments.done"));
    t.assert_true("completed",     has_event(done, "response.completed"));

    json response;
    for (const auto & event : done) {
        if (event.value("event", std::string()) == "response.completed") {
            response = event.at("data").at("response");
        }
    }
    t.assert_equal("output items", (size_t) 3, response.at("output").size());
    t.assert_equal("input usage",  4, response.at("usage").at("input_tokens").get<int>());
}

static void test_responses_non_stream(testing & t) {
    server_responses_store store;
    server_responses_request req;
    req.model = "m";
    server_responses_context ctx(store, req, "m", nullptr);

    server_task_result_cmpl_final result;
    result.n_prompt_tokens = 3;
    result.n_decoded       = 2;
    result.stop            = STOP_TYPE_EOS;
    result.truncated       = false;
    result.oaicompat_msg.role    = "assistant";
    result.oaicompat_msg.content = "hello";

    json response = ctx.final_response(result);
    t.assert_equal("status",       std::string("completed"), response.at("status").get<std::string>());
    t.assert_equal("output size",  (size_t) 1, response.at("output").size());
    t.assert_equal("output text",  std::string("hello"), response.at("output")[0].at("content")[0].at("text").get<std::string>());
    t.assert_equal("input usage",  3, response.at("usage").at("input_tokens").get<int>());

    server_responses_stored stored;
    t.assert_true("stored", store.get(ctx.get_id(), stored));
}


static void test_responses_tools_convert(testing & t) {
    json tools = json::array({
        json { {"type", "web_search"} },
        json {
            {"type", "function"},
            {"name", "get_weather"},
            {"description", "weather"},
            {"parameters", json::object()},
        },
    });

    json converted = server_chat_convert_responses_tools(tools);
    t.assert_equal("converted size", (size_t) 1, converted.size());
    t.assert_equal("converted type", std::string("function"), converted.at(0).at("type").get<std::string>());
    t.assert_equal("converted name", std::string("get_weather"), converted.at(0).at("function").at("name").get<std::string>());
    t.assert_true ("strict default", converted.at(0).at("function").at("strict").get<bool>());
}


//
// server_chat_completions
//

static void test_chat_completions_parse(testing & t) {
    json body = {
        {"model", "m"},
        {"messages", json::array({ json { {"role", "user"}, {"content", "hi"} } })},
        {"store", true},
        {"n", 2},
        {"metadata", json { {"k", "v"} }},
        {"service_tier", "default"},
    };
    server_chat_completions_request req = server_chat_completions_parse_request(body);
    t.assert_true ("store",     req.store);
    t.assert_equal("n",         2, req.n);
    t.assert_equal("user",      std::string(""), req.user);
    t.assert_equal("tier",      std::string("default"), req.service_tier);
    t.assert_true ("metadata",  !req.metadata.is_null());
    t.assert_equal("messages",  (size_t) 1, req.messages.size());

    bool thrown = false;
    json bad = body;
    bad["n"] = 0;
    try {
        server_chat_completions_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("n bound", thrown);

    thrown = false;
    bad = body;
    bad["metadata"] = json { {"k", 1} };
    try {
        server_chat_completions_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("metadata type", thrown);

    json normalize = {
        {"messages", json::array({ json { {"role", "user"}, {"content", "hi"} } })},
        {"tools", json::array({
            json { {"type", "function"}, {"function", json { {"name", "a"} }} },
            json { {"type", "function"}, {"function", json { {"name", "b"} }} },
        })},
        {"tool_choice", json {
            {"type", "function"},
            {"function", json { {"name", "a"} }},
        }},
    };
    server_chat_completions_normalize_request(normalize);
    t.assert_equal("filtered tools", (size_t) 1, normalize.at("tools").size());
    t.assert_equal("forced choice",  std::string("required"), normalize.at("tool_choice").get<std::string>());

    thrown = false;
    bad = body;
    bad["temperature"] = 3.0;
    try {
        server_chat_completions_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("temperature bound", thrown);

    thrown = false;
    bad = body;
    bad["stop"] = json::array({ "a", "b", "c", "d", "e" });
    try {
        server_chat_completions_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("stop count", thrown);

    thrown = false;
    bad = body;
    bad["tools"] = json::array({ json { {"type", "web_search"} } });
    try {
        server_chat_completions_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("web search rejected", thrown);

    thrown = false;
    bad = body;
    bad["service_tier"] = "ultra";
    try {
        server_chat_completions_parse_request(bad);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("service tier bound", thrown);

    json legacy = {
        {"messages", json::array({
            json { {"role", "function"}, {"name", "f"}, {"content", "x"} },
            json { {"role", "assistant"}, {"function_call", json { {"name", "f"}, {"arguments", "{}"} }} },
        })},
        {"functions", json::array({ json {
            {"name", "f"},
            {"description", "d"},
            {"parameters", json::object()},
        } })},
        {"function_call", json { {"name", "f"} }},
    };
    server_chat_completions_normalize_request(legacy);
    t.assert_true ("legacy functions converted", legacy.contains("tools") && !legacy.contains("functions"));
    t.assert_equal("legacy role", std::string("tool"), legacy.at("messages")[0].at("role").get<std::string>());
    t.assert_true ("legacy function_call converted", legacy.at("messages")[1].contains("tool_calls"));
    t.assert_true ("legacy choice converted", legacy.contains("tool_choice"));
}

static void test_chat_completions_store(testing & t) {
    server_chat_completions_store store;

    server_chat_completions_request req;
    req.model        = "m";
    req.store        = true;
    req.metadata     = json { {"k", "v"} };
    req.service_tier = "default";

    server_chat_completions_context ctx(store, req);

    json response = {
        {"id", "chatcmpl-test"},
        {"object", "chat.completion"},
        {"created", 1},
        {"model", "m"},
        {"choices", json::array({ json {
            {"index", 0},
            {"finish_reason", "stop"},
            {"message", json { {"role", "assistant"}, {"content", "hi"} }},
        } })},
        {"usage", json {
            {"prompt_tokens", 3},
            {"completion_tokens", 2},
            {"total_tokens", 5},
        }},
    };
    ctx.finish(response);

    t.assert_true("refusal",  response.at("choices")[0].at("message").contains("refusal"));
    t.assert_true("usage details", response.at("usage").contains("completion_tokens_details"));
    t.assert_true("metadata", response.contains("metadata"));

    server_chat_completions_stored stored;
    t.assert_true("stored", store.get("chatcmpl-test", stored));
    t.assert_equal("stored messages", (size_t) 1, stored.messages.size());
    t.assert_true("message id", stored.messages[0].value("id", std::string()).rfind("msg_", 0) == 0);

    t.assert_true("update metadata", store.update_metadata("chatcmpl-test", json { {"k", "v2"} }));
    t.assert_true("updated", store.get("chatcmpl-test", stored));
    t.assert_equal("updated value", std::string("v2"), stored.response.at("metadata").at("k").get<std::string>());
    t.assert_equal("list size", (size_t) 1, store.list().size());
    t.assert_true("erase", store.erase("chatcmpl-test"));

    // streaming merge: two choices must land in one stored response
    server_chat_completions_request stream_req;
    stream_req.store = true;
    stream_req.n     = 2;
    server_chat_completions_context stream_ctx(store, stream_req);

    json chunk0 = {
        {"id", "chatcmpl-stream"},
        {"model", "m"},
        {"choices", json::array({ json {
            {"index", 0},
            {"message", json { {"role", "assistant"}, {"content", "a"} }},
        } })},
        {"usage", json { {"prompt_tokens", 4}, {"completion_tokens", 1}, {"total_tokens", 5} }},
    };
    json chunk1 = chunk0;
    chunk1["choices"][0]["index"]   = 1;
    chunk1["choices"][0]["message"]["content"] = "b";
    stream_ctx.finish_stream(chunk0);
    t.assert_true("not stored before all choices", !store.get("chatcmpl-stream", stored));
    stream_ctx.finish_stream(chunk1);
    t.assert_true("stream stored", store.get("chatcmpl-stream", stored));
    t.assert_equal("merged choices", (size_t) 2, stored.response.at("choices").size());
    t.assert_equal("merged usage", 2, stored.response.at("usage").at("completion_tokens").get<int>());
}


static void test_openai_store_persistence(testing & t) {
    const std::filesystem::path dir = std::filesystem::current_path() / ("llama-openai-store-test-" + random_string());
    std::filesystem::remove_all(dir);

    server_responses_stored response;
    response.response    = json { {"id", "resp_test"}, {"object", "response"} };
    response.input_items = json::array({ json { {"role", "user"}, {"content", "hi"} } });
    response.created_at  = (int64_t) std::time(0);
    {
        server_responses_store store(dir.string(), 16, 0);
        store.put("resp_test", response);
    }
    {
        server_responses_store store(dir.string(), 16, 0);
        server_responses_stored out;
        t.assert_true ("responses persisted", store.get("resp_test", out));
        t.assert_equal("responses id",        std::string("resp_test"), out.response.at("id").get<std::string>());
        t.assert_true ("responses erase",     store.erase("resp_test"));
    }
    {
        server_responses_store store(dir.string(), 16, 0);
        server_responses_stored out;
        t.assert_true("responses erased", !store.get("resp_test", out));
    }

    server_chat_completions_stored chat;
    chat.response    = json { {"id", "chatcmpl_test"}, {"object", "chat.completion"} };
    chat.messages    = json::array({ json { {"id", "msg_1"}, {"role", "assistant"}, {"content", "hi"} } });
    chat.created_at  = (int64_t) std::time(0);
    {
        server_chat_completions_store store(dir.string(), 16, 0);
        store.put("chatcmpl_test", chat);
    }
    {
        server_chat_completions_store store(dir.string(), 16, 0);
        server_chat_completions_stored out;
        t.assert_true ("chat persisted", store.get("chatcmpl_test", out));
        t.assert_equal("chat messages",  (size_t) 1, out.messages.size());
        t.assert_true ("chat update",    store.update_metadata("chatcmpl_test", json { {"k", "v"} }));
        t.assert_equal("chat list",      (size_t) 1, store.list().size());
        t.assert_true ("chat erase",     store.erase("chatcmpl_test"));
    }

    server_responses_stored expired = response;
    expired.created_at = (int64_t) std::time(0) - 100;
    {
        server_responses_store store(dir.string(), 16, 0);
        store.put("resp_expired", expired);
    }
    {
        server_responses_store store(dir.string(), 16, 10);
        server_responses_stored out;
        t.assert_true("ttl expired", !store.get("resp_expired", out));
    }

    std::filesystem::remove_all(dir);
}


static void test_conversations_store(testing & t) {
    server_conversations_store store;

    server_conversation conversation = store.create(
        json { {"k", "v"} },
        json::array({ json { {"role", "user"}, {"content", "hello"} } }));

    t.assert_true ("conversation id", conversation.id.rfind("conv_", 0) == 0);
    t.assert_equal("conversation items", (size_t) 1, conversation.items.size());
    t.assert_true ("item id", conversation.items[0].value("id", std::string()).rfind("item_", 0) == 0);

    server_conversation got;
    t.assert_true("conversation get", store.get(conversation.id, got));

    json added = json::array();
    t.assert_true("conversation add", store.add_items(
        conversation.id,
        json::array({ json { {"role", "assistant"}, {"content", "hi"} } }),
        &added));
    t.assert_equal("added size", (size_t) 1, added.size());
    t.assert_true ("added id", added[0].value("id", std::string()).rfind("item_", 0) == 0);

    json data     = json::array();
    json first_id = nullptr;
    json last_id  = nullptr;
    bool has_more = false;
    t.assert_true("conversation list", store.list_items(conversation.id, "", 20, "asc", data, has_more, first_id, last_id));
    t.assert_equal("list size", (size_t) 2, data.size());
    t.assert_true ("list has no more", !has_more);

    json item;
    t.assert_true("conversation get item", store.get_item(conversation.id, added[0].value("id", std::string()), item));

    server_conversation after_delete;
    t.assert_true("conversation delete item",
        store.delete_item(conversation.id, added[0].value("id", std::string()), after_delete));
    t.assert_equal("items after delete", (size_t) 1, after_delete.items.size());

    t.assert_true("conversation update", store.update_metadata(conversation.id, json { {"k", "v2"} }));
    t.assert_true("conversation get after update", store.get(conversation.id, got));
    t.assert_equal("conversation metadata", std::string("v2"), got.metadata.at("k").get<std::string>());

    t.assert_true("conversation erase", store.erase(conversation.id));
    t.assert_true("conversation gone", !store.get(conversation.id, got));

    // Responses conversation input resolution
    server_responses_store responses_store;
    server_conversation ctx_conv = store.create(json(nullptr), json::array({ json { {"role", "user"}, {"content", "previous"} } }));
    json body = {
        {"model", "m"},
        {"input", "next"},
        {"conversation", ctx_conv.id},
    };
    server_responses_request req = server_responses_parse_request(body);
    t.assert_equal("conversation parsed", ctx_conv.id, req.conversation_id);
    server_responses_resolve_input(req, responses_store, &store);
    t.assert_equal("conversation context items", (size_t) 2, req.effective_input.size());
}


static void test_chat_completions_parallel(testing & t) {
    server_chat_completions_store store;
    server_chat_completions_request req;
    req.store                = false;
    req.parallel_tool_calls  = false;
    server_chat_completions_context ctx(store, req);

    json response = {
        {"id", "chatcmpl-parallel"},
        {"choices", json::array({ json {
            {"index", 0},
            {"message", json {
                {"role", "assistant"},
                {"content", ""},
                {"tool_calls", json::array({
                    json { {"id", "call_1"}, {"type", "function"}, {"function", json { {"name", "a"}, {"arguments", "{}"} }} },
                    json { {"id", "call_2"}, {"type", "function"}, {"function", json { {"name", "b"}, {"arguments", "{}"} }} },
                })},
            }},
        } })},
        {"usage", json { {"prompt_tokens", 1}, {"completion_tokens", 1}, {"total_tokens", 2} }},
    };
    ctx.finish(response);
    t.assert_equal("one tool call", (size_t) 1, response.at("choices")[0].at("message").at("tool_calls").size());

    json chunks = json::array({ json {
        {"choices", json::array({ json {
            {"delta", json {
                {"tool_calls", json::array({
                    json { {"index", 0}, {"id", "call_1"} },
                    json { {"index", 1}, {"id", "call_2"} },
                })},
            }},
        } })},
    } });
    json filtered = ctx.filter_chunks(chunks);
    t.assert_equal("one streamed tool call", (size_t) 1, filtered[0].at("choices")[0].at("delta").at("tool_calls").size());
}

static void test_responses_parallel(testing & t) {
    server_responses_store store;
    server_responses_request req;
    req.model                = "m";
    req.store                = false;
    req.parallel_tool_calls  = false;
    server_responses_context ctx(store, req, "m", nullptr);

    server_task_result_cmpl_final result;
    result.n_prompt_tokens = 1;
    result.n_decoded       = 2;
    result.stop            = STOP_TYPE_EOS;
    result.truncated       = false;

    common_chat_tool_call first;
    first.id = "call_1"; first.name = "a"; first.arguments = "{}";
    common_chat_tool_call second;
    second.id = "call_2"; second.name = "b"; second.arguments = "{}";
    result.oaicompat_msg.tool_calls = { first, second };

    json response = ctx.final_response(result);
    t.assert_equal("one response tool call", (size_t) 1, response.at("output").size());
    t.assert_equal("tool call type", std::string("function_call"), response.at("output")[0].at("type").get<std::string>());
}


static void test_responses_reasoning_summary(testing & t) {
    server_responses_store store;
    server_responses_request req;
    req.model    = "m";
    req.store    = false;
    req.reasoning = json { {"summary", "auto"} };
    server_responses_context ctx(store, req, "m", nullptr);

    common_chat_msg_diff diff;
    diff.reasoning_content_delta = "think";
    server_task_result_cmpl_partial partial;
    partial.oaicompat_msg_diffs = { diff };

    json events = ctx.consume(partial);
    t.assert_true("summary part added", has_event(events, "response.reasoning_summary_part.added"));
    t.assert_true("summary delta",      has_event(events, "response.reasoning_summary_text.delta"));
    t.assert_true("no reasoning text",  !has_event(events, "response.reasoning_text.delta"));

    server_task_result_cmpl_final final_result;
    final_result.n_prompt_tokens = 1;
    final_result.n_decoded       = 1;
    final_result.stop            = STOP_TYPE_EOS;
    final_result.truncated       = false;
    final_result.oaicompat_msg.reasoning_content = "think";

    json done = ctx.finish(final_result);
    t.assert_true("summary text done", has_event(done, "response.reasoning_summary_text.done"));
    t.assert_true("summary part done", has_event(done, "response.reasoning_summary_part.done"));
    t.assert_true("completed",         has_event(done, "response.completed"));

    json response;
    for (const auto & event : done) {
        if (event.value("event", std::string()) == "response.completed") {
            response = event.at("data").at("response");
        }
    }
    t.assert_equal("reasoning item", (size_t) 1, response.at("output").size());
    t.assert_equal("summary size",   (size_t) 1, response.at("output")[0].at("summary").size());
}


static void test_responses_logprobs(testing & t) {
    server_responses_store store;
    server_responses_request req;
    req.model            = "m";
    req.store            = false;
    req.include_logprobs = true;
    req.top_logprobs     = 1;
    server_responses_context ctx(store, req, "m", nullptr);

    server_task_result_cmpl_final result;
    result.n_prompt_tokens = 2;
    result.n_decoded       = 1;
    result.stop            = STOP_TYPE_EOS;
    result.truncated       = false;
    result.oaicompat_msg.role    = "assistant";
    result.oaicompat_msg.content = "hi";

    completion_token_output token;
    token.tok           = 1;
    token.prob          = 0.5f;
    token.text_to_send  = "hi";
    token.probs.push_back({ 1, "hi", 0.5f });
    result.probs_output.push_back(token);

    json response = ctx.final_response(result);
    const json & logprobs = response.at("output")[0].at("content")[0].at("logprobs");
    t.assert_equal("logprob count", (size_t) 1, logprobs.size());
    t.assert_equal("logprob token", std::string("hi"), logprobs[0].at("token").get<std::string>());
    t.assert_true ("logprob negative", logprobs[0].at("logprob").get<double>() < 0.0);
}


static void test_responses_failed_event(testing & t) {
    server_responses_store store;
    server_responses_request req;
    req.model = "m";
    req.store = true;
    server_responses_context ctx(store, req, "m", nullptr);

    json event = ctx.failed_event("boom", "server_error");
    t.assert_equal("failed event", std::string("response.failed"), event.at("event").get<std::string>());
    t.assert_equal("failed status", std::string("failed"), event.at("data").at("response").at("status").get<std::string>());
    t.assert_equal("failed message", std::string("boom"), event.at("data").at("response").at("error").at("message").get<std::string>());

    server_responses_stored stored;
    t.assert_true("failed stored", store.get(ctx.get_id(), stored));
}


static void test_openai_error_shape(testing & t) {
    const std::string message = "'temperature' is out of range";
    const json err = format_openai_error(message, ERROR_TYPE_INVALID_REQUEST, "", json("temperature"));
    t.assert_equal("http status", 400, err.at("http_status").get<int>());
    t.assert_equal("type",        std::string("invalid_request_error"), err.at("type").get<std::string>());
    t.assert_equal("param",       std::string("temperature"), err.at("param").get<std::string>());
    t.assert_true ("code null",   err.at("code").is_null());
    t.assert_equal("extracted",   std::string("temperature"), error_param_from_message(message));
}


static void test_files_store(testing & t) {
    const std::string dir = (std::filesystem::temp_directory_path() / ("llama-files-" + random_string())).string();
    {
        server_files_store store(dir, 8, 2);
        server_file file;
        std::string error;
        const raw_buffer data = {'h', 'e', 'l', 'l', 'o'};

        t.assert_true ("create", store.create("hello.txt", "assistants", "text/plain", data, file, error));
        t.assert_equal("bytes", 5, file.bytes);
        t.assert_equal("filename", std::string("hello.txt"), file.filename);

        server_file loaded;
        t.assert_true("get", store.get(file.id, loaded));
        raw_buffer content;
        t.assert_true("content", store.content(file.id, content));
        t.assert_equal("content bytes", 5, (int) content.size());

        server_files_store reopened(dir, 8, 2);
        t.assert_true("reopened get", reopened.get(file.id, loaded));

        const raw_buffer too_big(9, 'x');
        server_file rejected;
        t.assert_true("size limit", !store.create("big.bin", "assistants", "application/octet-stream", too_big, rejected, error));

        server_file second;
        const raw_buffer one = {'x'};
        t.assert_true("second", store.create("two.txt", "assistants", "text/plain", one, second, error));
        const raw_buffer two = {'y', 'y'};
        t.assert_true("entry limit", !store.create("three.txt", "assistants", "text/plain", two, rejected, error));

        std::vector<server_file> files;
        bool has_more = false;
        t.assert_true("list", store.list("assistants", "", 1, false, files, has_more));
        t.assert_equal("list size", 1, (int) files.size());
        t.assert_true("has more", has_more);

        t.assert_true("remove", store.remove(file.id, loaded));
        t.assert_true("removed get", !store.get(file.id, loaded));
    }
    std::filesystem::remove_all(dir);
}


static void test_files_http(testing & t) {
    server_files_store store("", 16, 4);
    std::function<bool()> stop = [] { return false; };

    server_http_req create_req{{}, {}, "/v1/files", "", "{\"purpose\":\"assistants\"}", {}, stop};
    create_req.files["file"] = uploaded_file{{'h', 'i'}, "hi.txt", "text/plain"};
    auto created = server_files_http::create(store, create_req);
    t.assert_equal("create status", 200, created->status);

    json created_json = json::parse(created->data);
    const std::string id = created_json.value("id", std::string());
    t.assert_true("file id", !id.empty());
    t.assert_equal("file object", std::string("file"), created_json.value("object", std::string()));

    server_http_req list_req{{}, {}, "/v1/files", "", "", {}, stop};
    list_req.params["purpose"] = "assistants";
    auto listed = server_files_http::list(store, list_req);
    t.assert_equal("list status", 200, listed->status);
    json listed_json = json::parse(listed->data);
    t.assert_equal("list size", 1, (int) listed_json.at("data").size());

    server_http_req get_req{{}, {}, "/v1/files/" + id, "", "", {}, stop};
    get_req.params["id"] = id;
    auto got = server_files_http::get(store, get_req);
    t.assert_equal("get status", 200, got->status);

    server_http_req content_req = get_req;
    content_req.path = "/v1/files/" + id + "/content";
    auto content = server_files_http::content(store, content_req);
    t.assert_equal("content status", 200, content->status);
    t.assert_equal("content body", std::string("hi"), content->data);

    auto deleted = server_files_http::del(store, get_req);
    t.assert_equal("delete status", 200, deleted->status);
    auto missing = server_files_http::get(store, get_req);
    t.assert_equal("missing status", 404, missing->status);

    server_http_req bad_create = create_req;
    bad_create.files.clear();
    t.assert_equal("missing file status", 400, server_files_http::create(store, bad_create)->status);
}


static void test_responses_custom_tools(testing & t) {
    json body = {
        {"model", "m"},
        {"input", "hi"},
        {"tools", json::array({ json {
            {"type", "custom"},
            {"name", "run_code"},
            {"description", "Run code"},
            {"format", json {
                {"type", "grammar"},
                {"grammar", json {
                    {"syntax", "lark"},
                    {"definition", "start: /.+/"},
                }},
            }},
        } })},
        {"tool_choice", json {
            {"type", "custom"},
            {"name", "run_code"},
        }},
    };

    server_responses_request req = server_responses_parse_request(body);
    t.assert_equal("custom tools", (size_t) 1, req.tools_parsed.size());
    t.assert_true ("is custom", req.tools_parsed.at(0).is_custom);
    t.assert_equal("custom name", std::string("run_code"), req.tools_parsed.at(0).name);
    t.assert_equal("custom format", std::string("grammar"), req.tools_parsed.at(0).custom_format.type);
    t.assert_equal("custom syntax", std::string("lark"), req.tools_parsed.at(0).custom_format.syntax);
    t.assert_equal("custom choice", std::string("run_code"), req.tool_choice_name);

    bool thrown = false;
    body["tools"] = json::array({ json {
        {"type", "custom"},
        {"name", "bad"},
        {"format", json {{"type", "json"}}},
    } });
    try {
        server_responses_parse_request(body);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("custom format rejected", thrown);
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("server_tokens", test_server_tokens);
    t.test("utf8",          test_utf8);
    t.test("sse",           test_sse);
    t.test("json_helpers",  test_json_helpers);
    t.test("lora_helpers",  test_lora_helpers);
    t.test("server_pipe",   test_server_pipe);
    t.test("decision_escape", test_decision_escape);
    t.test("rerank_template", test_rerank_template);
    t.test("decision_class_scores", test_decision_class_scores);
    t.test("decision_ocr_jev_validation", test_decision_ocr_jev_validation);
    t.test("decision_depth",  test_decision_depth);
    t.test("glob_match",      test_glob_match);
    t.test("dry_sequence_breakers", test_dry_sequence_breakers);
    t.test("speech_convert",  test_speech_convert);
    t.test("pcm_f32_to_s16",  test_pcm_f32_to_s16);
    t.test("responses_parse",     test_responses_parse);
    t.test("responses_previous",  test_responses_previous);
    t.test("responses_stream",    test_responses_stream);
    t.test("responses_non_stream", test_responses_non_stream);
    t.test("responses_tools_convert", test_responses_tools_convert);
    t.test("responses_custom_tools", test_responses_custom_tools);
    t.test("chat_completions_parse",  test_chat_completions_parse);
    t.test("chat_completions_store",  test_chat_completions_store);
    t.test("openai_store_persistence", test_openai_store_persistence);
    t.test("conversations_store",     test_conversations_store);
    t.test("chat_completions_parallel", test_chat_completions_parallel);
    t.test("responses_parallel",        test_responses_parallel);
    t.test("responses_reasoning_summary", test_responses_reasoning_summary);
    t.test("responses_logprobs",        test_responses_logprobs);
    t.test("responses_failed_event",    test_responses_failed_event);
    t.test("openai_error_shape",        test_openai_error_shape);
    t.test("files_store",               test_files_store);
    t.test("files_http",                test_files_http);

    return t.summary();
}
