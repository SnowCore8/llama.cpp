#include "server-openai-store.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace server_openai_store {

int64_t now() {
    return (int64_t) std::time(0);
}

static bool valid_id(const std::string & id) {
    if (id.empty() || id == "." || id == "..") {
        return false;
    }
    for (const char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
        if (!ok) {
            return false;
        }
    }
    return true;
}

std::string item_path(const std::string & base, const std::string & subdir, const std::string & id) {
    if (base.empty() || !valid_id(id)) {
        return std::string();
    }
    return (fs::path(base) / subdir / (id + ".json")).string();
}

std::vector<std::string> list_files(const std::string & dir) {
    std::vector<std::string> files;
    std::error_code ec;
    if (!fs::is_directory(dir, ec) || ec) {
        return files;
    }
    for (const auto & entry : fs::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            files.push_back(entry.path().string());
        }
    }
    return files;
}

bool read_json(const std::string & path, json & data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    try {
        data = json::parse(buffer.str());
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

bool write_json(const std::string & path, const json & data) {
    const fs::path file_path(path);
    std::error_code ec;
    fs::create_directories(file_path.parent_path(), ec);
    if (ec) {
        return false;
    }
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            return false;
        }
        file << data.dump();
        if (!file) {
            return false;
        }
    }
    fs::rename(tmp_path, file_path, ec);
    if (ec) {
        fs::remove(file_path, ec);
        ec.clear();
        fs::rename(tmp_path, file_path, ec);
    }
    return !ec;
}

bool remove_file(const std::string & path) {
    std::error_code ec;
    if (path.empty()) {
        return false;
    }
    return fs::remove(path, ec) && !ec;
}

} // namespace server_openai_store
