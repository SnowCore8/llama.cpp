#include "server-web-search.h"

#include "server-common.h"
#include "server-responses.h"
#include "download.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_set>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#endif

namespace fs = std::filesystem;

namespace {

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace((unsigned char) s.front())) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace((unsigned char) s.back())) {
        s.pop_back();
    }
    return s;
}

std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

bool sockaddr_is_private(const sockaddr * sa) {
    if (sa == nullptr) {
        return true;
    }
    if (sa->sa_family == AF_INET) {
        const uint32_t a = ntohl(((const sockaddr_in *) sa)->sin_addr.s_addr);
        const uint8_t b1 = (uint8_t) (a >> 24);
        const uint8_t b2 = (uint8_t) ((a >> 16) & 0xff);
        if (b1 == 0 || b1 == 10 || b1 == 127) {
            return true;
        }
        if (b1 == 169 && b2 == 254) {
            return true;
        }
        if (b1 == 172 && b2 >= 16 && b2 <= 31) {
            return true;
        }
        if (b1 == 192 && b2 == 168) {
            return true;
        }
        if (b1 == 100 && b2 >= 64 && b2 <= 127) {
            return true; // CGNAT
        }
        return false;
    }
    if (sa->sa_family == AF_INET6) {
        const uint8_t * b = ((const sockaddr_in6 *) sa)->sin6_addr.s6_addr;
        bool all_zero = true;
        for (int i = 0; i < 16; ++i) {
            if (b[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            return true;
        }
        bool loop = true;
        for (int i = 0; i < 15; ++i) {
            if (b[i] != 0) {
                loop = false;
                break;
            }
        }
        if (loop && b[15] == 1) {
            return true;
        }
        if ((b[0] & 0xfe) == 0xfc) {
            return true; // ULA fc00::/7
        }
        if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) {
            return true; // link-local
        }
        bool v4mapped = true;
        for (int i = 0; i < 10; ++i) {
            if (b[i] != 0) {
                v4mapped = false;
                break;
            }
        }
        if (v4mapped && b[10] == 0xff && b[11] == 0xff) {
            sockaddr_in v4;
            std::memset(&v4, 0, sizeof(v4));
            v4.sin_family = AF_INET;
            std::memcpy(&v4.sin_addr, b + 12, 4);
            return sockaddr_is_private((const sockaddr *) &v4);
        }
        return false;
    }
    return true;
}

std::string url_encode(const std::string & s) {
    static const char * hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size() * 3);
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back((char) c);
        } else if (c == ' ') {
            out.push_back('+');
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0x0F]);
        }
    }
    return out;
}

std::string url_decode(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+' ) {
            out.push_back(' ');
        } else if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            const int hi = hex(s[i + 1]);
            const int lo = hex(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back((char) ((hi << 4) | lo));
                i += 2;
                continue;
            }
            out.push_back(s[i]);
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::string html_unescape(std::string s) {
    static const std::pair<const char *, const char *> ents[] = {
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""},
        {"&#39;", "'"}, {"&apos;", "'"}, {"&nbsp;", " "},
    };
    for (const auto & e : ents) {
        size_t pos = 0;
        const std::string from = e.first;
        const std::string to = e.second;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    }
    return s;
}

std::string strip_tags(std::string s) {
    static const std::regex re_tag("<[^>]*>");
    s = std::regex_replace(s, re_tag, " ");
    s = html_unescape(s);
    static const std::regex re_ws("\\s+");
    s = std::regex_replace(s, re_ws, " ");
    return trim_copy(s);
}

std::string host_of_url(const std::string & url) {
    static const std::regex re(R"(^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]+))");
    std::smatch m;
    if (!std::regex_search(url, m, re)) {
        return {};
    }
    std::string host = m[1].str();
    // strip userinfo / port
    auto at = host.find('@');
    if (at != std::string::npos) {
        host = host.substr(at + 1);
    }
    if (!host.empty() && host.front() == '[') {
        auto end = host.find(']');
        if (end != std::string::npos) {
            return to_lower(host.substr(1, end - 1));
        }
    }
    auto colon = host.find(':');
    if (colon != std::string::npos) {
        host = host.substr(0, colon);
    }
    return to_lower(host);
}

bool domain_allowed(
        const std::string & url,
        const std::vector<std::string> & allowed,
        const std::vector<std::string> & blocked) {
    const std::string host = host_of_url(url);
    if (host.empty() && url.rfind("about:", 0) == 0) {
        return allowed.empty();
    }
    auto matches = [&](const std::string & rule) {
        const std::string r = to_lower(rule);
        if (r.empty()) {
            return false;
        }
        return host == r || (host.size() > r.size() && host.compare(host.size() - r.size() - 1, r.size() + 1, "." + r) == 0);
    };
    for (const auto & b : blocked) {
        if (matches(b)) {
            return false;
        }
    }
    if (allowed.empty()) {
        return true;
    }
    for (const auto & a : allowed) {
        if (matches(a)) {
            return true;
        }
    }
    return false;
}

void push_unique_result(json & results, json item, size_t max_results) {
    if (results.size() >= max_results) {
        return;
    }
    const std::string url = json_value(item, "url", std::string());
    if (!url.empty()) {
        for (const auto & existing : results) {
            if (json_value(existing, "url", std::string()) == url) {
                return;
            }
        }
    }
    results.push_back(std::move(item));
}

std::string extract_last_user_text(const json & body) {
    auto text_from_content = [](const json & content) -> std::string {
        if (content.is_string()) {
            return content.get<std::string>();
        }
        if (!content.is_array()) {
            return {};
        }
        std::string out;
        for (const auto & part : content) {
            if (!part.is_object()) {
                continue;
            }
            const std::string t = json_value(part, "type", std::string());
            if (t == "text" || t == "input_text" || t == "output_text" || t.empty()) {
                if (part.contains("text") && part.at("text").is_string()) {
                    if (!out.empty()) {
                        out.push_back('\n');
                    }
                    out += part.at("text").get<std::string>();
                }
            }
        }
        return out;
    };

    if (body.contains("input")) {
        const auto & input = body.at("input");
        if (input.is_string()) {
            return input.get<std::string>();
        }
        if (input.is_array()) {
            for (size_t i = input.size(); i-- > 0;) {
                const auto & item = input.at(i);
                if (!item.is_object()) {
                    continue;
                }
                const std::string role = json_value(item, "role", std::string());
                const std::string type = json_value(item, "type", std::string());
                if (role == "user" || type == "message") {
                    if (item.contains("content")) {
                        std::string t = text_from_content(item.at("content"));
                        if (!t.empty()) {
                            return t;
                        }
                    }
                }
            }
        }
    }
    if (body.contains("messages") && body.at("messages").is_array()) {
        const auto & messages = body.at("messages");
        for (size_t i = messages.size(); i-- > 0;) {
            const auto & msg = messages.at(i);
            if (!msg.is_object()) {
                continue;
            }
            if (json_value(msg, "role", std::string()) != "user") {
                continue;
            }
            if (msg.contains("content")) {
                std::string t = text_from_content(msg.at("content"));
                if (!t.empty()) {
                    return t;
                }
            }
        }
    }
    return {};
}

void parse_domain_list(const json & arr, std::vector<std::string> & out) {
    if (!arr.is_array()) {
        return;
    }
    for (const auto & v : arr) {
        if (v.is_string()) {
            std::string d = trim_copy(v.get<std::string>());
            if (!d.empty()) {
                out.push_back(std::move(d));
            }
        }
    }
}

void merge_filters_from_tool(const json & tool, server_web_search_options & opt) {
    if (tool.contains("search_context_size") && tool.at("search_context_size").is_string()) {
        opt.context_size = tool.at("search_context_size").get<std::string>();
    }
    if (tool.contains("external_web_access") && tool.at("external_web_access").is_boolean()) {
        opt.external_web_access = tool.at("external_web_access").get<bool>();
    }
    if (tool.contains("filters") && tool.at("filters").is_object()) {
        const auto & f = tool.at("filters");
        if (f.contains("allowed_domains")) {
            parse_domain_list(f.at("allowed_domains"), opt.allowed_domains);
        }
        if (f.contains("blocked_domains")) {
            parse_domain_list(f.at("blocked_domains"), opt.blocked_domains);
        }
    }
    if (tool.contains("user_location") && tool.at("user_location").is_object()) {
        const auto & ul = tool.at("user_location");
        std::ostringstream oss;
        for (const char * k : {"country", "region", "city", "timezone"}) {
            if (ul.contains(k) && ul.at(k).is_string()) {
                if (oss.tellp() > 0) {
                    oss << ", ";
                }
                oss << ul.at(k).get<std::string>();
            }
        }
        opt.user_location = oss.str();
    }
}

void apply_context_size(server_web_search_options & opt) {
    const std::string cs = to_lower(opt.context_size);
    if (cs == "low") {
        opt.max_results = std::min(opt.max_results, 3);
        opt.fetch_pages = false;
        opt.fetch_pages_n = 0;
    } else if (cs == "high") {
        opt.max_results = std::max(opt.max_results, 8);
        opt.fetch_pages = true;
        opt.fetch_pages_n = std::max(opt.fetch_pages_n, 3);
    } else {
        // medium default
        opt.max_results = std::max(3, std::min(opt.max_results, 5));
        opt.fetch_pages = true;
        opt.fetch_pages_n = std::max(opt.fetch_pages_n, 1);
    }
}

std::vector<std::string> build_queries(const server_web_search_options & opt) {
    std::vector<std::string> queries;
    std::string base = trim_copy(opt.explicit_query.empty() ? opt.query : opt.explicit_query);
    if (base.empty()) {
        base = "status";
    }
    if (base.size() > 512) {
        base = base.substr(0, 512);
    }
    if (!opt.user_location.empty()) {
        base += " location:" + opt.user_location;
    }
    queries.push_back(base);
    return queries;
}

json remote_get_json_or_text(const std::string & url, long timeout_s) {
    common_remote_params params;
    params.timeout = timeout_s;
    auto remote = common_remote_get_content(url, params);
    json out = {
        {"http_code", remote.first},
        {"body",      std::string(remote.second.begin(), remote.second.end())},
    };
    return out;
}

void parse_ddg_instant(const json & payload, json & results, size_t max_results,
                       const server_web_search_options & opt) {
    auto push_item = [&](const std::string & title, const std::string & link, const std::string & snippet) {
        if (title.empty() && snippet.empty() && link.empty()) {
            return;
        }
        if (!link.empty() && !domain_allowed(link, opt.allowed_domains, opt.blocked_domains)) {
            return;
        }
        push_unique_result(results, json{
            {"title",   title.empty() ? link : title},
            {"url",     link},
            {"snippet", snippet},
            {"source",  "duckduckgo_ia"},
        }, max_results);
    };

    const std::string abs = json_value(payload, "AbstractText", std::string());
    const std::string abs_url = json_value(payload, "AbstractURL", std::string());
    const std::string heading = json_value(payload, "Heading", std::string());
    if (!abs.empty()) {
        push_item(heading.empty() ? "Abstract" : heading, abs_url, abs);
    }
    const std::string answer = json_value(payload, "Answer", std::string());
    if (!answer.empty()) {
        push_item(heading.empty() ? "Answer" : heading, abs_url, answer);
    }
    const std::string def = json_value(payload, "Definition", std::string());
    if (!def.empty()) {
        push_item("Definition", json_value(payload, "DefinitionURL", std::string()), def);
    }
    if (payload.contains("RelatedTopics") && payload.at("RelatedTopics").is_array()) {
        for (const auto & topic : payload.at("RelatedTopics")) {
            if (!topic.is_object()) {
                continue;
            }
            if (topic.contains("Topics") && topic.at("Topics").is_array()) {
                for (const auto & nested : topic.at("Topics")) {
                    if (!nested.is_object()) {
                        continue;
                    }
                    push_item(
                        json_value(nested, "Text", std::string()),
                        json_value(nested, "FirstURL", std::string()),
                        json_value(nested, "Text", std::string()));
                }
            } else {
                push_item(
                    json_value(topic, "Text", std::string()),
                    json_value(topic, "FirstURL", std::string()),
                    json_value(topic, "Text", std::string()));
            }
        }
    }
    if (payload.contains("Results") && payload.at("Results").is_array()) {
        for (const auto & item : payload.at("Results")) {
            if (!item.is_object()) {
                continue;
            }
            push_item(
                json_value(item, "Text", std::string()),
                json_value(item, "FirstURL", std::string()),
                json_value(item, "Text", std::string()));
        }
    }
}

void parse_generic_results(const json & payload, json & results, size_t max_results,
                           const server_web_search_options & opt, const char * source) {
    if (!payload.contains("results") || !payload.at("results").is_array()) {
        return;
    }
    for (const auto & item : payload.at("results")) {
        if (!item.is_object()) {
            continue;
        }
        const std::string url = json_value(item, "url", json_value(item, "FirstURL", std::string()));
        if (!url.empty() && !domain_allowed(url, opt.allowed_domains, opt.blocked_domains)) {
            continue;
        }
        push_unique_result(results, json{
            {"title",   json_value(item, "title", json_value(item, "Text", url))},
            {"url",     url},
            {"snippet", json_value(item, "snippet", json_value(item, "Text", std::string()))},
            {"source",  source},
        }, max_results);
    }
}

void parse_ddg_html(const std::string & html, json & results, size_t max_results,
                    const server_web_search_options & opt) {
    // DuckDuckGo HTML: result links often contain uddg=<urlencoded url>
    static const std::regex re_uddg(R"re(uddg=([^&"']+))re");
    static const std::regex re_result_a(
        R"re(<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>)re");
    static const std::regex re_snippet(
        R"re(<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)</a>|<td class="result-snippet">([\s\S]*?)</td>)re");

    std::sregex_iterator it(html.begin(), html.end(), re_result_a);
    std::sregex_iterator end;
    std::vector<std::pair<std::string, std::string>> links;
    for (; it != end && links.size() < max_results * 2; ++it) {
        std::string href = (*it)[1].str();
        std::string title = strip_tags((*it)[2].str());
        std::smatch um;
        if (std::regex_search(href, um, re_uddg)) {
            href = url_decode(um[1].str());
        } else if (href.rfind("//", 0) == 0) {
            href = "https:" + href;
        } else if (href.rfind("/", 0) == 0) {
            continue;
        }
        links.emplace_back(href, title);
    }

    // lite.duckduckgo.com uses simpler anchors
    if (links.empty()) {
        static const std::regex re_lite(R"re(<a rel="nofollow" href="(https?://[^"]+)"[^>]*>([\s\S]*?)</a>)re");
        std::sregex_iterator lit(html.begin(), html.end(), re_lite);
        for (; lit != end && links.size() < max_results * 2; ++lit) {
            links.emplace_back((*lit)[1].str(), strip_tags((*lit)[2].str()));
        }
    }

    std::vector<std::string> snippets;
    std::sregex_iterator sit(html.begin(), html.end(), re_snippet);
    for (; sit != end && snippets.size() < max_results * 2; ++sit) {
        std::string sn = (*sit)[1].matched ? (*sit)[1].str() : (*sit)[2].str();
        snippets.push_back(strip_tags(sn));
    }

    for (size_t i = 0; i < links.size() && results.size() < max_results; ++i) {
        const std::string & url = links[i].first;
        if (!domain_allowed(url, opt.allowed_domains, opt.blocked_domains)) {
            continue;
        }
        std::string snip = i < snippets.size() ? snippets[i] : std::string();
        push_unique_result(results, json{
            {"title",   links[i].second.empty() ? url : links[i].second},
            {"url",     url},
            {"snippet", snip},
            {"source",  "duckduckgo_html"},
        }, max_results);
    }
}

void search_local_dir(const std::string & query, json & results, size_t max_results) {
    const char * dir = std::getenv("LLAMA_WEB_SEARCH_LOCAL_DIR");
    if (!dir || dir[0] == '\0' || !fs::exists(dir)) {
        return;
    }
    std::vector<std::string> terms;
    {
        std::istringstream iss(to_lower(query));
        std::string w;
        while (iss >> w) {
            if (w.size() >= 3) {
                terms.push_back(w);
            }
        }
    }
    if (terms.empty()) {
        return;
    }

    struct hit {
        int score;
        std::string path;
        std::string snippet;
    };
    std::vector<hit> hits;

    for (auto const & entry : fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto ext = entry.path().extension().string();
        if (ext != ".txt" && ext != ".md" && ext != ".markdown" && ext != ".html" && ext != ".json") {
            continue;
        }
        std::ifstream in(entry.path());
        if (!in) {
            continue;
        }
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (content.size() > 256 * 1024) {
            content.resize(256 * 1024);
        }
        const std::string low = to_lower(content);
        int score = 0;
        size_t first_pos = std::string::npos;
        for (const auto & t : terms) {
            size_t pos = low.find(t);
            if (pos != std::string::npos) {
                score += 1;
                first_pos = std::min(first_pos, pos);
            }
        }
        if (score == 0) {
            continue;
        }
        size_t start = first_pos == std::string::npos ? 0 : (first_pos > 80 ? first_pos - 80 : 0);
        std::string snip = content.substr(start, std::min<size_t>(240, content.size() - start));
        snip = strip_tags(snip);
        hits.push_back({score, entry.path().string(), snip});
    }
    std::sort(hits.begin(), hits.end(), [](const hit & a, const hit & b) { return a.score > b.score; });
    for (const auto & h : hits) {
        push_unique_result(results, json{
            {"title",   fs::path(h.path).filename().string()},
            {"url",     std::string("file://") + h.path},
            {"snippet", h.snippet},
            {"source",  "local_dir"},
        }, max_results);
    }
}

void load_fixture(json & results, size_t max_results, const server_web_search_options & opt) {
    const char * path = std::getenv("LLAMA_WEB_SEARCH_FIXTURE");
    if (!path || path[0] == '\0' || !fs::exists(path)) {
        return;
    }
    try {
        std::ifstream in(path);
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        json payload = json::parse_no_throw(content);
        if (payload.is_discarded()) {
            return;
        }
        parse_generic_results(payload, results, max_results, opt, "fixture");
        if (results.empty() && payload.is_array()) {
            for (const auto & item : payload) {
                if (!item.is_object()) {
                    continue;
                }
                const std::string url = json_value(item, "url", std::string());
                if (!url.empty() && !domain_allowed(url, opt.allowed_domains, opt.blocked_domains)) {
                    continue;
                }
                push_unique_result(results, json{
                    {"title",   json_value(item, "title", url)},
                    {"url",     url},
                    {"snippet", json_value(item, "snippet", std::string())},
                    {"source",  "fixture"},
                }, max_results);
            }
        }
    } catch (...) {
        LOG_WRN("%s", "LLAMA_WEB_SEARCH_FIXTURE parse failed\n");
    }
}

std::string fetch_page_text(const std::string & url) {
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
        return {};
    }
    // block obvious loopback / private targets (SSRF)
    auto host_of = [](const std::string & u, bool lower) -> std::string {
        size_t start = u.find("://");
        if (start == std::string::npos) {
            return {};
        }
        start += 3;
        size_t end = u.find_first_of("/?#", start);
        std::string hostport = (end == std::string::npos) ? u.substr(start) : u.substr(start, end - start);
        if (!hostport.empty() && hostport.front() == '[') {
            size_t rb = hostport.find(']');
            if (rb != std::string::npos) {
                std::string h = hostport.substr(1, rb - 1);
                return lower ? to_lower(h) : h;
            }
        }
        size_t colon = hostport.rfind(':');
        if (colon != std::string::npos && hostport.find(':') == colon) {
            hostport = hostport.substr(0, colon);
        }
        return lower ? to_lower(hostport) : hostport;
    };
    const std::string host = host_of(url, false);
    const std::string host_l = to_lower(host);
    auto is_private_host = [](const std::string & h) -> bool {
        if (h.empty() || h == "localhost" || h == "127.0.0.1" || h == "0.0.0.0" ||
            h == "::1" || h == "0:0:0:0:0:0:0:1") {
            return true;
        }
        if (h.rfind("127.", 0) == 0 || h.rfind("10.", 0) == 0 || h.rfind("192.168.", 0) == 0 ||
            h.rfind("169.254.", 0) == 0) {
            return true;
        }
        if (h.find(':') != std::string::npos) {
            // IPv6 ULA / link-local
            if (h.rfind("fc", 0) == 0 || h.rfind("fd", 0) == 0 || h.rfind("fe80", 0) == 0) {
                return true;
            }
        }
        if (h.rfind("172.", 0) == 0 && h.size() >= 6) {
            // 172.16.0.0/12
            size_t d1 = h.find('.', 4);
            if (d1 != std::string::npos) {
                try {
                    int second = std::stoi(h.substr(4, d1 - 4));
                    if (second >= 16 && second <= 31) {
                        return true;
                    }
                } catch (...) {}
            }
        }
        return false;
    };
    // Resolve once, reject any private answer, pin that IP for the HTTP(S) connect
    // (Host/SNI stay on `host`) so a second DNS lookup cannot rebind to LAN.
    auto resolve_public_ip = [&](const std::string & h, std::string & out_ip) -> bool {
        if (h.empty() || is_private_host(h)) {
            return false;
        }
        addrinfo hints;
        std::memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo * res = nullptr;
        if (getaddrinfo(h.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) {
            return false;
        }
        bool ok = true;
        std::string first;
        for (addrinfo * p = res; p != nullptr; p = p->ai_next) {
            if (sockaddr_is_private(p->ai_addr)) {
                ok = false;
                break;
            }
            if (first.empty() && p->ai_addr != nullptr) {
                char buf[INET6_ADDRSTRLEN];
                if (p->ai_family == AF_INET) {
                    if (inet_ntop(AF_INET, &((sockaddr_in *) p->ai_addr)->sin_addr, buf, sizeof(buf))) {
                        first = buf;
                    }
                } else if (p->ai_family == AF_INET6) {
                    if (inet_ntop(AF_INET6, &((sockaddr_in6 *) p->ai_addr)->sin6_addr, buf, sizeof(buf))) {
                        first = buf;
                    }
                }
            }
        }
        freeaddrinfo(res);
        if (!ok || first.empty()) {
            return false;
        }
        out_ip = first;
        return true;
    };
    if (is_private_host(host_l)) {
        return {};
    }
    std::string pinned_ip;
    if (!resolve_public_ip(host_l, pinned_ip)) {
        return {};
    }
    try {
        common_remote_params params;
        params.timeout = 5;
        params.follow_location = false; // redirects could re-target private hosts
        // Pin using the same host string httplib parses from the URL (case-preserving).
        params.hostname_addr_map[host] = pinned_ip;
        auto remote = common_remote_get_content(url, params);
        json remote_j = {
            {"http_code", remote.first},
            {"body",      std::string(remote.second.begin(), remote.second.end())},
        };
        if (json_value(remote_j, "http_code", 0L) < 200 || json_value(remote_j, "http_code", 0L) >= 400) {
            return {};
        }
        std::string body = json_value(remote_j, "body", std::string());
        // Prefer <p> text blocks
        static const std::regex re_p("<p[^>]*>([\\s\\S]*?)</p>", std::regex::icase);
        std::ostringstream oss;
        std::sregex_iterator it(body.begin(), body.end(), re_p);
        std::sregex_iterator end;
        for (; it != end && oss.tellp() < 3000; ++it) {
            std::string p = strip_tags((*it)[1].str());
            if (p.size() < 40) {
                continue;
            }
            oss << p << " ";
        }
        std::string out = trim_copy(oss.str());
        if (out.empty()) {
            out = strip_tags(body);
        }
        if (out.size() > 2500) {
            out = out.substr(0, 2500);
        }
        return out;
    } catch (...) {
        return {};
    }
}

std::vector<std::string> tokenize_words(const std::string & text) {
    std::vector<std::string> words;
    std::string cur;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            cur.push_back((char) std::tolower(c));
        } else if (!cur.empty()) {
            if (cur.size() >= 3) {
                words.push_back(cur);
            }
            cur.clear();
        }
    }
    if (cur.size() >= 3) {
        words.push_back(cur);
    }
    return words;
}

int overlap_score(const std::string & text, const json & result) {
    const auto words = tokenize_words(text);
    if (words.empty()) {
        return 0;
    }
    std::unordered_set<std::string> bag(words.begin(), words.end());
    const std::string blob = to_lower(
        json_value(result, "title", std::string()) + " " +
        json_value(result, "snippet", std::string()) + " " +
        json_value(result, "page_text", std::string()));
    int score = 0;
    for (const auto & w : bag) {
        if (blob.find(w) != std::string::npos) {
            score++;
        }
    }
    return score;
}

} // namespace

bool server_web_search_is_tool_type(const std::string & type) {
    if (type == "web_search" || type == "web_search_preview") {
        return true;
    }
    return false;
}

server_web_search_options server_web_search_options_from_body(const json & body) {
    server_web_search_options opt;
    opt.query = extract_last_user_text(body);

    if (body.contains("web_search_options") && body.at("web_search_options").is_object()) {
        const auto & wso = body.at("web_search_options");
        // Honor Chat-shaped options: search_context_size, filters.*, user_location, etc.
        merge_filters_from_tool(wso, opt);
        if (wso.contains("query") && wso.at("query").is_string()) {
            opt.explicit_query = wso.at("query").get<std::string>();
        }
        // OpenAI approximate location nest: {type, approximate:{country,city,...}}
        if (wso.contains("user_location") && wso.at("user_location").is_object()) {
            const auto & ul = wso.at("user_location");
            if (ul.contains("approximate") && ul.at("approximate").is_object()) {
                merge_filters_from_tool(json{{"user_location", ul.at("approximate")}}, opt);
            }
        }
    }

    if (body.contains("tools") && body.at("tools").is_array()) {
        for (const auto & tool : body.at("tools")) {
            if (!tool.is_object()) {
                continue;
            }
            const std::string type = json_value(tool, "type", std::string());
            if (!server_web_search_is_tool_type(type)) {
                continue;
            }
            merge_filters_from_tool(tool, opt);
        }
    }

    // Echo tools kept from prior strip
    if (body.contains("__oai_web_search_echo_tools") && body.at("__oai_web_search_echo_tools").is_array()) {
        for (const auto & tool : body.at("__oai_web_search_echo_tools")) {
            if (tool.is_object()) {
                merge_filters_from_tool(tool, opt);
            }
        }
    }

    apply_context_size(opt);
    return opt;
}

json server_web_search_run(const server_web_search_options & opt_in) {
    server_web_search_options opt = opt_in;
    apply_context_size(opt);
    const auto queries = build_queries(opt);

    json results = json::array();
    json actions = json::array();
    std::string provider = "none";
    int n_requests = 0;

    // Fixture first (deterministic offline completeness)
    load_fixture(results, (size_t) opt.max_results, opt);
    if (!results.empty()) {
        provider = "fixture";
    }

    search_local_dir(queries.front(), results, (size_t) opt.max_results);
    if (provider == "none" && !results.empty()) {
        provider = "local_dir";
    }

    auto try_custom_and_remote = [&](const std::string & q) {
        if (!opt.external_web_access) {
            return;
        }
        std::vector<std::string> urls;
        if (const char * env = std::getenv("LLAMA_WEB_SEARCH_URL")) {
            if (env[0] != '\0') {
                std::string url = env;
                const std::string enc = url_encode(q);
                for (const char * token : {"{q}", "%s"}) {
                    const std::string tok = token;
                    size_t pos = 0;
                    while ((pos = url.find(tok, pos)) != std::string::npos) {
                        url.replace(pos, tok.size(), enc);
                        pos += enc.size();
                    }
                }
                urls.push_back(url);
            }
        }
        // Instant Answer + HTML SERP
        urls.push_back(
            "https://api.duckduckgo.com/?q=" + url_encode(q) +
            "&format=json&no_html=1&skip_disambig=1");
        urls.push_back("https://html.duckduckgo.com/html/?q=" + url_encode(q));
        urls.push_back("https://lite.duckduckgo.com/lite/?q=" + url_encode(q));

        for (const auto & url : urls) {
            if ((int) results.size() >= opt.max_results) {
                break;
            }
            try {
                auto remote = remote_get_json_or_text(url, 6);
                n_requests++;
                const long code = json_value(remote, "http_code", 0L);
                const std::string body = json_value(remote, "body", std::string());
                if (code < 200 || code >= 300 || body.empty()) {
                    continue;
                }
                if (body.size() > 0 && (body[0] == '{' || body[0] == '[')) {
                    json payload = json::parse_no_throw(body);
                    if (!payload.is_discarded()) {
                        const size_t before = results.size();
                        parse_generic_results(payload, results, (size_t) opt.max_results, opt, "custom");
                        if (results.size() > before) {
                            provider = "custom";
                            continue;
                        }
                        parse_ddg_instant(payload, results, (size_t) opt.max_results, opt);
                        if (results.size() > before) {
                            provider = "duckduckgo";
                            continue;
                        }
                    }
                }
                const size_t before = results.size();
                parse_ddg_html(body, results, (size_t) opt.max_results, opt);
                if (results.size() > before) {
                    provider = "duckduckgo_html";
                }
            } catch (const std::exception & e) {
                LOG_WRN("web_search fetch failed url=%s err=%s\n", url.c_str(), e.what());
            } catch (...) {
                LOG_WRN("web_search fetch failed url=%s\n", url.c_str());
            }
        }
    };

    for (const auto & q : queries) {
        // Prefer offline fixture/local hits. Once fixture/local_dir produced any
        // results, skip remote SERP (avoids long HTTPS timeouts in --offline labs
        // even when result count is below max_results).
        const bool offline_hits =
            provider == "fixture" || provider == "local_dir";
        if (!offline_hits && (int) results.size() < opt.max_results) {
            try_custom_and_remote(q);
        }
        json sources_for_action = json::array();
        // Include all current result URLs (fixture/local may prefill before remote).
        for (const auto & item : results) {
            const std::string url = json_value(item, "url", std::string());
            if (url.empty()) {
                continue;
            }
            sources_for_action.push_back(json{
                {"type", "url"},
                {"url",  url},
            });
        }
        actions.push_back(json{
            {"type",    "search"},
            {"query",   q},
            {"queries", json::array({q})},
            {"sources", sources_for_action},
        });
        if ((int) results.size() >= opt.max_results) {
            break;
        }
    }

    // open_page / find_in_page deepen (skip when results are purely offline fixture/local_dir
    // to avoid long HTTPS timeouts in offline labs).
    const bool allow_page_fetch =
        opt.fetch_pages && opt.fetch_pages_n > 0 && opt.external_web_access &&
        provider != "fixture" && provider != "local_dir" && provider != "none";
    if (allow_page_fetch) {
        int fetched = 0;
        for (auto & item : results) {
            if (fetched >= opt.fetch_pages_n) {
                break;
            }
            const std::string url = json_value(item, "url", std::string());
            if (url.rfind("http", 0) != 0) {
                continue;
            }
            std::string page = fetch_page_text(url);
            n_requests++;
            if (page.empty()) {
                continue;
            }
            item["page_text"] = page;
            if (json_value(item, "snippet", std::string()).empty()) {
                item["snippet"] = page.substr(0, std::min<size_t>(280, page.size()));
            }
            actions.push_back(json{
                {"type", "open_page"},
                {"url",  url},
            });
            // find_in_page: first significant query term
            const auto words = tokenize_words(queries.front());
            for (const auto & w : words) {
                if (to_lower(page).find(w) != std::string::npos) {
                    actions.push_back(json{
                        {"type",    "find_in_page"},
                        {"url",     url},
                        {"pattern", w},
                    });
                    break;
                }
            }
            fetched++;
        }
    }

    if (results.empty()) {
        // No fabricated hits — empty results are honest offline/no-match semantics.
        provider = "none";
        if (actions.empty()) {
            actions.push_back(json{
                {"type",    "search"},
                {"query",   queries.front()},
                {"queries", json::array({queries.front()})},
                {"sources", json::array()},
            });
        }
    }

    return json{
        {"query",      queries.front()},
        {"queries",    queries},
        {"results",    results},
        {"provider",   provider},
        {"actions",    actions},
        {"n_requests", n_requests},
    };
}

void server_web_search_apply(json & body) {
    if (json_value(body, "__oai_web_search", false)) {
        return;
    }

    bool want = false;
    if (body.contains("web_search_options") && !body.at("web_search_options").is_null()) {
        want = true;
    }
    json kept_tools = json::array();
    json echo_web_tools = json::array();
    if (body.contains("tools") && body.at("tools").is_array()) {
        for (const auto & tool : body.at("tools")) {
            if (!tool.is_object()) {
                kept_tools.push_back(tool);
                continue;
            }
            const std::string type = json_value(tool, "type", std::string());
            if (server_web_search_is_tool_type(type)) {
                want = true;
                echo_web_tools.push_back(tool);
                continue;
            }
            kept_tools.push_back(tool);
        }
        if (want) {
            body["tools"] = kept_tools;
        }
    }
    if (!want) {
        return;
    }

    if (!echo_web_tools.empty()) {
        body["__oai_web_search_echo_tools"] = echo_web_tools;
    }

    server_web_search_options opt = server_web_search_options_from_body(body);
    json packed = server_web_search_run(opt);

    body["__oai_web_search"] = true;
    body["__oai_web_search_query"] = packed.at("query");
    body["__oai_web_search_results"] = packed.at("results");
    body["__oai_web_search_actions"] = packed.at("actions");
    body["__oai_web_search_n_requests"] = packed.at("n_requests");

    std::ostringstream ctx;
    ctx << "You have local web_search results. Prefer these over parametric memory when relevant.\n";
    ctx << "provider=" << packed.at("provider").get<std::string>()
        << " context_size=" << opt.context_size << "\n";
    ctx << "queries:\n";
    for (const auto & q : packed.at("queries")) {
        if (q.is_string()) {
            ctx << "- " << q.get<std::string>() << "\n";
        }
    }
    int i = 1;
    for (const auto & item : packed.at("results")) {
        ctx << i << ". " << json_value(item, "title", std::string()) << "\n";
        const std::string link = json_value(item, "url", std::string());
        if (!link.empty()) {
            ctx << "   URL: " << link << "\n";
        }
        const std::string snip = json_value(item, "snippet", std::string());
        if (!snip.empty()) {
            ctx << "   Snippet: " << snip << "\n";
        }
        const std::string page = json_value(item, "page_text", std::string());
        if (!page.empty()) {
            ctx << "   Page: " << page.substr(0, std::min<size_t>(900, page.size())) << "\n";
        }
        i++;
        if (i > opt.max_results + 1) {
            break;
        }
    }
    ctx << "Cite sources with markdown links like [title](url) when you use them.";
    const std::string block = ctx.str();

    if (body.contains("instructions") || body.contains("input")) {
        if (body.contains("instructions") && body.at("instructions").is_string()) {
            body["instructions"] = body.at("instructions").get<std::string>() + "\n\n" + block;
        } else {
            body["instructions"] = block;
        }
    }

    if (body.contains("messages") && body.at("messages").is_array()) {
        json & messages = body.at("messages");
        if (!messages.empty() && messages[0].is_object() &&
                json_value(messages[0], "role", std::string()) == "system") {
            if (messages[0].contains("content") && messages[0].at("content").is_string()) {
                messages[0]["content"] =
                    messages[0].at("content").get<std::string>() + "\n\n" + block;
            } else {
                messages[0]["content"] = block;
            }
        } else {
            json new_messages = json::array();
            new_messages.push_back({{"role", "system"}, {"content", block}});
            for (const auto & m : messages) {
                new_messages.push_back(m);
            }
            messages = std::move(new_messages);
        }
    }

    LOG_INF("web_search applied query='%s' provider=%s results=%zu actions=%zu n_requests=%d\n",
            packed.at("query").get<std::string>().c_str(),
            packed.at("provider").get<std::string>().c_str(),
            packed.at("results").size(),
            packed.at("actions").size(),
            packed.at("n_requests").get<int>());
}

json server_web_search_responses_output_items(const json & request_body) {
    json out = json::array();
    if (!json_value(request_body, "__oai_web_search", false)) {
        return out;
    }
    const bool want_sources =
        server_responses_include_contains(request_body, "web_search_call.action.sources");
    const bool want_results =
        server_responses_include_contains(request_body, "web_search_call.results");

    json actions = json::array();
    if (request_body.contains("__oai_web_search_actions") &&
            request_body.at("__oai_web_search_actions").is_array()) {
        actions = request_body.at("__oai_web_search_actions");
    } else {
        actions.push_back(json{
            {"type",  "search"},
            {"query", json_value(request_body, "__oai_web_search_query", std::string())},
        });
    }

    for (const auto & action_in : actions) {
        if (!action_in.is_object()) {
            continue;
        }
        json action = action_in;
        const std::string atype = json_value(action, "type", std::string("search"));
        if (atype == "search") {
            if (!want_sources) {
                action.erase("sources");
            } else {
                bool empty_sources = !action.contains("sources") ||
                    !action.at("sources").is_array() || action.at("sources").empty();
                if (empty_sources && request_body.contains("__oai_web_search_results") &&
                        request_body.at("__oai_web_search_results").is_array()) {
                    json sources = json::array();
                    for (const auto & item : request_body.at("__oai_web_search_results")) {
                        sources.push_back(json{
                            {"type", "url"},
                            {"url",  json_value(item, "url", std::string())},
                        });
                    }
                    action["sources"] = sources;
                }
            }
        }
        json item = {
            {"id",     "ws_" + random_string()},
            {"type",   "web_search_call"},
            {"status", "completed"},
            {"action", action},
        };
        if (want_results && atype == "search" &&
                request_body.contains("__oai_web_search_results")) {
            item["results"] = request_body.at("__oai_web_search_results");
        }
        out.push_back(std::move(item));
    }
    return out;
}

void server_web_search_annotate_responses_output(json & response_obj, const json & request_body) {
    if (!json_value(request_body, "__oai_web_search", false)) {
        return;
    }
    if (!response_obj.contains("output") || !response_obj.at("output").is_array()) {
        return;
    }
    if (!request_body.contains("__oai_web_search_results") ||
            !request_body.at("__oai_web_search_results").is_array()) {
        return;
    }
    const json & results = request_body.at("__oai_web_search_results");

    for (auto & item : response_obj.at("output")) {
        if (!item.is_object() || json_value(item, "type", std::string()) != "message") {
            continue;
        }
        if (!item.contains("content") || !item.at("content").is_array()) {
            continue;
        }
        for (auto & part : item.at("content")) {
            if (!part.is_object() || json_value(part, "type", std::string()) != "output_text") {
                continue;
            }
            if (!part.contains("text") || !part.at("text").is_string()) {
                continue;
            }
            std::string text = part.at("text").get<std::string>();
            json annotations = part.contains("annotations") && part.at("annotations").is_array()
                                   ? part.at("annotations")
                                   : json::array();

            // Prefer markdown links already present
            static const std::regex re_md(R"(\[([^\]]+)\]\((https?://[^)]+)\))");
            bool any = false;
            for (std::sregex_iterator it(text.begin(), text.end(), re_md), end; it != end; ++it) {
                annotations.push_back(json{
                    {"type",        "url_citation"},
                    {"start_index", (int) it->position(0)},
                    {"end_index",   (int) (it->position(0) + it->length(0))},
                    {"url",         (*it)[2].str()},
                    {"title",       (*it)[1].str()},
                });
                any = true;
            }

            if (!any) {
                // Score results against answer; cite top overlaps, else top 2
                std::vector<std::pair<int, size_t>> ranked;
                for (size_t i = 0; i < results.size(); ++i) {
                    ranked.push_back({overlap_score(text, results[i]), i});
                }
                std::sort(ranked.begin(), ranked.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
                size_t n_cite = 0;
                for (const auto & r : ranked) {
                    if (r.first <= 0 && n_cite >= 2) {
                        break;
                    }
                    if (n_cite >= 3) {
                        break;
                    }
                    if (r.first <= 0 && n_cite > 0 && ranked.front().first > 0) {
                        break;
                    }
                    const auto & src = results[r.second];
                    const std::string url = json_value(src, "url", std::string());
                    const std::string title = json_value(src, "title", url);
                    if (url.empty() || url.rfind("about:", 0) == 0) {
                        continue;
                    }
                    const std::string marker = " [" + std::to_string(n_cite + 1) + "]";
                    const int start = (int) text.size();
                    text += marker;
                    const int end = (int) text.size();
                    annotations.push_back(json{
                        {"type",        "url_citation"},
                        {"start_index", start},
                        {"end_index",   end},
                        {"url",         url},
                        {"title",       title},
                    });
                    n_cite++;
                    any = true;
                }
                if (any) {
                    text += "\n";
                    for (size_t i = 0; i < annotations.size(); ++i) {
                        if (json_value(annotations[i], "type", std::string()) != "url_citation") {
                            continue;
                        }
                        // only newly appended numeric markers — rewrite Sources footer once
                    }
                    text += "Sources:";
                    int idx = 1;
                    for (const auto & ann : annotations) {
                        if (json_value(ann, "type", std::string()) != "url_citation") {
                            continue;
                        }
                        text += "\n[" + std::to_string(idx++) + "] " +
                                json_value(ann, "title", std::string()) + " — " +
                                json_value(ann, "url", std::string());
                    }
                }
            }

            part["text"] = text;
            part["annotations"] = annotations;
        }
    }

    // refresh convenience field
    if (response_obj.contains("output_text")) {
        std::string text;
        for (const auto & item : response_obj.at("output")) {
            if (json_value(item, "type", std::string()) != "message") {
                continue;
            }
            if (!item.contains("content") || !item.at("content").is_array()) {
                continue;
            }
            for (const auto & part : item.at("content")) {
                if (json_value(part, "type", std::string()) == "output_text" &&
                        part.contains("text") && part.at("text").is_string()) {
                    text += part.at("text").get<std::string>();
                }
            }
        }
        response_obj["output_text"] = text;
    }
}

void server_web_search_annotate_chat_message(json & message_obj, const json & web_results) {
    if (!message_obj.is_object() || !web_results.is_array()) {
        return;
    }
    std::string text;
    if (message_obj.contains("content") && message_obj.at("content").is_string()) {
        text = message_obj.at("content").get<std::string>();
    }
    json annotations = json::array();
    if (web_results.empty()) {
        message_obj["annotations"] = annotations;
        return;
    }
    std::vector<std::pair<int, size_t>> ranked;
    for (size_t i = 0; i < web_results.size(); ++i) {
        ranked.push_back({overlap_score(text, web_results[i]), i});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });
    size_t n = 0;
    for (const auto & r : ranked) {
        if (n >= 3) {
            break;
        }
        if (r.first <= 0 && n >= 1) {
            break;
        }
        const auto & src = web_results[r.second];
        const std::string url = json_value(src, "url", std::string());
        if (url.empty() || url.rfind("about:", 0) == 0) {
            continue;
        }
        annotations.push_back(json{
            {"type",        "url_citation"},
            {"url",         url},
            {"title",       json_value(src, "title", url)},
            {"start_index", 0},
            {"end_index",   0},
        });
        n++;
    }
    // Always set when annotate is invoked (search ran); empty list if filters dropped all hits.
    message_obj["annotations"] = annotations;
}
