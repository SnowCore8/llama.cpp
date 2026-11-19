#pragma once

#include "server-common.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// OpenAI Files API object backed by local storage or memory
struct server_file {
    std::string id;
    std::string filename;
    std::string purpose;
    std::string content_type;
    std::string status = "processed";
    int64_t bytes = 0;
    int64_t created_at = 0;
};

json server_file_public_json(const server_file & file);

// local file store: metadata and bytes under <base>/files, memory only when base is empty
struct server_files_store {
    server_files_store(std::string base_path, size_t max_file_bytes, size_t max_entries);

    // create a file from an uploaded buffer, fills out and returns false on validation/storage error
    bool create(
            const std::string & filename,
            const std::string & purpose,
            const std::string & content_type,
            const raw_buffer & data,
            server_file & out,
            std::string & error);

    bool get(const std::string & id, server_file & out) const;
    bool content(const std::string & id, raw_buffer & out) const;
    bool remove(const std::string & id, server_file & out);
    bool list(
            const std::string & purpose,
            const std::string & after,
            size_t limit,
            bool ascending,
            std::vector<server_file> & out,
            bool & has_more) const;

private:
    std::string base_path;
    size_t      max_file_bytes;
    size_t      max_entries;

    mutable std::mutex mutex;
    // used when base_path is empty
    std::map<std::string, server_file> memory_files;
    std::map<std::string, raw_buffer>  memory_content;

    bool is_memory() const { return base_path.empty(); }
    std::string content_path(const std::string & id) const;
    bool load_locked(const std::string & id, server_file & out) const;
    bool save_locked(const server_file & file, const raw_buffer & data);
    size_t count_locked() const;
};