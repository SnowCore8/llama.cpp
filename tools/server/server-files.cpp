#include "server-files.h"

#include "server-openai-store.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

json server_file_public_json(const server_file & file) {
    return json {
        {"id",             file.id},
        {"object",         "file"},
        {"bytes",          file.bytes},
        {"created_at",     file.created_at},
        {"filename",       file.filename},
        {"purpose",        file.purpose},
        {"status",         file.status},
        {"status_details", nullptr},
    };
}

static bool write_binary(const std::string & path, const raw_buffer & data) {
    const fs::path file_path(path);
    std::error_code ec;
    fs::create_directories(file_path.parent_path(), ec);
    if (ec) {
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return false;
    }
    if (!data.empty()) {
        file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
    }
    return (bool) file;
}

static bool read_binary(const std::string & path, raw_buffer & data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    data.resize((size_t) size);
    if (!data.empty()) {
        file.read(reinterpret_cast<char *>(data.data()), size);
    }
    return (bool) file;
}

server_files_store::server_files_store(std::string base_path, size_t max_file_bytes, size_t max_entries) :
        base_path(std::move(base_path)),
        max_file_bytes(max_file_bytes),
        max_entries(max_entries) {
    if (!is_memory()) {
        std::error_code ec;
        fs::create_directories(fs::path(this->base_path) / "files", ec);
    }
}

std::string server_files_store::content_path(const std::string & id) const {
    const std::string meta = server_openai_store::item_path(base_path, "files", id);
    if (meta.empty()) {
        return std::string();
    }
    return meta.substr(0, meta.size() - 5) + ".bin";
}

bool server_files_store::load_locked(const std::string & id, server_file & out) const {
    if (id.empty()) {
        return false;
    }
    if (is_memory()) {
        const auto it = memory_files.find(id);
        if (it == memory_files.end()) {
            return false;
        }
        out = it->second;
        return true;
    }
    json data;
    if (!server_openai_store::read_json(server_openai_store::item_path(base_path, "files", id), data)) {
        return false;
    }
    out.id           = data.value("id", std::string());
    out.filename     = data.value("filename", std::string());
    out.purpose      = data.value("purpose", std::string());
    out.content_type = data.value("content_type", std::string());
    out.status       = data.value("status", std::string("processed"));
    out.bytes        = data.value("bytes", (int64_t) 0);
    out.created_at   = data.value("created_at", (int64_t) 0);
    return !out.id.empty();
}

size_t server_files_store::count_locked() const {
    if (is_memory()) {
        return memory_files.size();
    }
    return server_openai_store::list_files((fs::path(base_path) / "files").string()).size();
}

bool server_files_store::save_locked(const server_file & file, const raw_buffer & data) {
    if (!write_binary(content_path(file.id), data)) {
        return false;
    }
    const json meta {
        {"id",             file.id},
        {"filename",       file.filename},
        {"purpose",        file.purpose},
        {"content_type",   file.content_type},
        {"status",         file.status},
        {"bytes",          file.bytes},
        {"created_at",     file.created_at},
    };
    if (!server_openai_store::write_json(server_openai_store::item_path(base_path, "files", file.id), meta)) {
        std::error_code ec;
        fs::remove(content_path(file.id), ec);
        return false;
    }
    return true;
}

bool server_files_store::create(
        const std::string & filename,
        const std::string & purpose,
        const std::string & content_type,
        const raw_buffer & data,
        server_file & out,
        std::string & error) {
    if (filename.empty()) {
        error = "'file' must have a filename";
        return false;
    }
    if (purpose.empty()) {
        error = "'purpose' is required";
        return false;
    }
    if (max_file_bytes > 0 && data.size() > max_file_bytes) {
        error = "file exceeds the maximum size";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (max_entries > 0 && count_locked() >= max_entries) {
        error = "file limit reached";
        return false;
    }

    server_file file;
    file.id           = "file-" + random_string();
    file.filename     = filename.substr(0, std::min<size_t>(filename.size(), 255));
    file.purpose      = purpose;
    file.content_type = content_type;
    file.bytes        = (int64_t) data.size();
    file.created_at   = server_openai_store::now();

    if (is_memory()) {
        memory_files[file.id] = file;
        memory_content[file.id] = data;
    } else if (!save_locked(file, data)) {
        error = "failed to persist file";
        return false;
    }
    out = file;
    return true;
}

bool server_files_store::get(const std::string & id, server_file & out) const {
    std::lock_guard<std::mutex> lock(mutex);
    return load_locked(id, out);
}

bool server_files_store::content(const std::string & id, raw_buffer & out) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (is_memory()) {
        const auto it = memory_content.find(id);
        if (it == memory_content.end()) {
            return false;
        }
        out = it->second;
        return true;
    }
    server_file file;
    return load_locked(id, file) && read_binary(content_path(id), out);
}

bool server_files_store::remove(const std::string & id, server_file & out) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!load_locked(id, out)) {
        return false;
    }
    if (is_memory()) {
        memory_content.erase(id);
        memory_files.erase(id);
    } else {
        std::error_code ec;
        fs::remove(content_path(id), ec);
        fs::remove(server_openai_store::item_path(base_path, "files", id), ec);
    }
    return true;
}

bool server_files_store::list(
        const std::string & purpose,
        const std::string & after,
        size_t limit,
        bool ascending,
        std::vector<server_file> & out,
        bool & has_more) const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<server_file> files;
    if (is_memory()) {
        for (const auto & [id, file] : memory_files) {
            files.push_back(file);
        }
    } else {
        const std::string dir = (fs::path(base_path) / "files").string();
        for (const auto & path : server_openai_store::list_files(dir)) {
            json data;
            if (!server_openai_store::read_json(path, data)) {
                continue;
            }
            server_file file;
            const std::string id = data.value("id", std::string());
            if (id.empty()) {
                continue;
            }
            file.id           = id;
            file.filename     = data.value("filename", std::string());
            file.purpose      = data.value("purpose", std::string());
            file.content_type = data.value("content_type", std::string());
            file.status       = data.value("status", std::string("processed"));
            file.bytes        = data.value("bytes", (int64_t) 0);
            file.created_at   = data.value("created_at", (int64_t) 0);
            files.push_back(std::move(file));
        }
    }

    files.erase(std::remove_if(files.begin(), files.end(), [&](const server_file & file) {
        return !purpose.empty() && file.purpose != purpose;
    }), files.end());
    std::sort(files.begin(), files.end(), [](const server_file & a, const server_file & b) {
        if (a.created_at != b.created_at) {
            return a.created_at < b.created_at;
        }
        return a.id < b.id;
    });
    if (!ascending) {
        std::reverse(files.begin(), files.end());
    }

    size_t begin = 0;
    if (!after.empty()) {
        auto it = std::find_if(files.begin(), files.end(), [&](const server_file & file) {
            return file.id == after;
        });
        if (it != files.end()) {
            begin = (size_t) (it - files.begin()) + 1;
        } else {
            begin = files.size();
        }
    }
    const size_t end = limit == 0 ? files.size() : std::min(files.size(), begin + limit);
    out.assign(files.begin() + (std::ptrdiff_t) begin, files.begin() + (std::ptrdiff_t) end);
    has_more = end < files.size();
    return true;
}