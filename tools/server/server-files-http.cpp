#include "server-files-http.h"

#include "server-common.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace server_files_http {

static server_http_res_ptr make_error(int status, const std::string & message, const std::string & code, const std::string & param = "") {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    json error = format_openai_error(message, ERROR_TYPE_INVALID_REQUEST, code,
        param.empty() ? json(nullptr) : json(param));
    error.erase("http_status");
    res->data = safe_json_to_str(json {{"error", error}});
    return res;
}

static server_http_res_ptr make_json(const json & data) {
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(data);
    return res;
}

static bool valid_purpose(const std::string & purpose) {
    static const std::set<std::string> purposes = {
        "assistants", "batch", "fine-tune", "vision", "user_data",
    };
    return purposes.count(purpose) > 0;
}

server_http_res_ptr create(server_files_store & store, const server_http_req & req) {
    const auto it = req.files.find("file");
    if (it == req.files.end()) {
        return make_error(400, "'file' is required", "missing_required_parameter", "file");
    }

    json body;
    try {
        body = req.body.empty() ? json::object() : json::parse(req.body);
    } catch (const std::exception &) {
        return make_error(400, "request body must be a JSON object", "invalid_request_error");
    }
    const std::string purpose = json_value(body, "purpose", std::string());
    if (purpose.empty()) {
        return make_error(400, "'purpose' is required", "missing_required_parameter", "purpose");
    }
    if (!valid_purpose(purpose)) {
        return make_error(400, "unsupported 'purpose' value: " + purpose, "invalid_value", "purpose");
    }

    server_file file;
    std::string error;
    if (!store.create(it->second.filename, purpose, it->second.content_type, it->second.data, file, error)) {
        return make_error(400, error, "invalid_request_error");
    }
    return make_json(server_file_public_json(file));
}

server_http_res_ptr list(server_files_store & store, const server_http_req & req) {
    const std::string purpose = req.get_param("purpose");
    if (!purpose.empty() && !valid_purpose(purpose)) {
        return make_error(400, "unsupported 'purpose' value: " + purpose, "invalid_value", "purpose");
    }
    const std::string order = req.get_param("order", "desc");
    if (order != "asc" && order != "desc") {
        return make_error(400, "'order' must be 'asc' or 'desc'", "invalid_value", "order");
    }
    size_t limit = 10000;
    const std::string limit_param = req.get_param("limit");
    if (!limit_param.empty()) {
        try {
            const long long value = std::stoll(limit_param);
            if (value < 1 || value > 10000) {
                return make_error(400, "'limit' must be between 1 and 10000", "invalid_value", "limit");
            }
            limit = (size_t) value;
        } catch (const std::exception &) {
            return make_error(400, "'limit' must be an integer", "invalid_value", "limit");
        }
    }

    std::vector<server_file> files;
    bool has_more = false;
    store.list(purpose, req.get_param("after"), limit, order == "asc", files, has_more);

    json data = json::array();
    for (const auto & file : files) {
        data.push_back(server_file_public_json(file));
    }
    const json first_id = files.empty() ? json(nullptr) : json(files.front().id);
    const json last_id  = files.empty() ? json(nullptr) : json(files.back().id);
    return make_json(json {
        {"object",   "list"},
        {"data",     data},
        {"first_id", first_id},
        {"last_id",  last_id},
        {"has_more", has_more},
    });
}

server_http_res_ptr get(server_files_store & store, const server_http_req & req) {
    server_file file;
    if (!store.get(req.get_param("id"), file)) {
        return make_error(404, "File not found", "file_not_found");
    }
    return make_json(server_file_public_json(file));
}

server_http_res_ptr del(server_files_store & store, const server_http_req & req) {
    server_file file;
    const std::string id = req.get_param("id");
    if (!store.remove(id, file)) {
        return make_error(404, "File not found", "file_not_found");
    }
    return make_json(json {
        {"id",      id},
        {"object",  "file"},
        {"deleted", true},
    });
}

server_http_res_ptr content(server_files_store & store, const server_http_req & req) {
    const std::string id = req.get_param("id");
    server_file file;
    raw_buffer data;
    if (!store.get(id, file) || !store.content(id, data)) {
        return make_error(404, "File not found", "file_not_found");
    }
    auto res = std::make_unique<server_http_res>();
    res->content_type = file.content_type.empty() ? "application/octet-stream" : file.content_type;
    std::string safe_filename;
    for (const char c : file.filename) {
        if (c != '"' && c != '\r' && c != '\n') {
            safe_filename += c;
        }
    }
    res->headers["Content-Disposition"] = "attachment; filename=\"" + safe_filename + "\"";
    res->data.assign(reinterpret_cast<const char *>(data.data()), data.size());
    return res;
}

} // namespace server_files_http