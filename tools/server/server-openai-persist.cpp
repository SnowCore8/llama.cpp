#include "server-openai-persist.h"
#include "log.h"

#include <filesystem>
#include <fstream>
#include <system_error>

namespace fs = std::filesystem;

namespace openai_persist {

static std::string g_root;

void set_root(const std::string & root) {
    g_root = root;
}

const std::string & root() {
    return g_root;
}

std::string join_dir(const std::string & root, const std::string & sub) {
    if (root.empty()) {
        return "";
    }
    std::string out = root;
    if (out.back() != DIRECTORY_SEPARATOR) {
        out += DIRECTORY_SEPARATOR;
    }
    if (!sub.empty()) {
        out += sub;
        if (out.back() != DIRECTORY_SEPARATOR) {
            out += DIRECTORY_SEPARATOR;
        }
    }
    return out;
}

bool ensure_dir(const std::string & dir) {
    if (dir.empty()) {
        return false;
    }
    std::error_code ec;
    fs::create_directories(dir, ec);
    return !ec && fs::is_directory(dir, ec);
}

static bool safe_id_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '_' || c == '-';
}

std::string safe_id_legacy(const std::string & id) {
    // Old encoding: map unsafe bytes to '_'. Distinct ids could collide.
    std::string out;
    out.reserve(id.size());
    for (unsigned char c : id) {
        if (safe_id_char(c)) {
            out.push_back((char) c);
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '_') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    if (out.empty()) {
        out = "id";
    }
    return out;
}

std::string safe_id(const std::string & id) {
    bool all_safe = !id.empty();
    for (unsigned char c : id) {
        if (!safe_id_char(c)) {
            all_safe = false;
            break;
        }
    }
    if (all_safe) {
        return id;
    }
    static const char * hex = "0123456789abcdef";
    std::string out = "h_";
    out.reserve(2 + id.size() * 2);
    for (unsigned char c : id) {
        out.push_back(hex[c >> 4]);
        out.push_back(hex[c & 0xf]);
    }
    if (out.size() == 2) {
        out += "00";
    }
    return out;
}

std::string json_path(const std::string & dir, const std::string & id) {
    return dir + safe_id(id) + ".json";
}

bool write_json_file(const std::string & path, const json & obj) {
    try {
        const fs::path p(path);
        if (p.has_parent_path()) {
            std::error_code ec;
            fs::create_directories(p.parent_path(), ec);
        }
        const std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out) {
                return false;
            }
            out << obj.dump();
        }
        std::error_code ec;
        fs::rename(tmp, path, ec);
        if (ec) {
            fs::remove(path, ec);
            fs::rename(tmp, path, ec);
        }
        return !ec;
    } catch (...) {
        return false;
    }
}

bool read_json_file(const std::string & path, json & out) {
    try {
        std::ifstream in(path);
        if (!in) {
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        out = json::parse(content);
        return true;
    } catch (...) {
        return false;
    }
}

void erase_file(const std::string & path) {
    std::error_code ec;
    fs::remove(path, ec);
}

std::vector<std::string> list_json_basenames(const std::string & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return out;
    }
    for (const auto & ent : fs::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!ent.is_regular_file()) {
            continue;
        }
        if (ent.path().extension() != ".json") {
            continue;
        }
        out.push_back(ent.path().filename().string());
    }
    return out;
}

} // namespace openai_persist
