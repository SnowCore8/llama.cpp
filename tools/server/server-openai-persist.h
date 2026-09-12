// Shared disk persistence helpers for durable OpenAI object stores
#pragma once

#include "common.h"

#include "server-common.h"

#include <string>
#include <vector>


namespace openai_persist {

// Root dir for durable local objects (responses, chat_completions, prompts,
// prompt_cache_keys). Empty = memory-only. Set once at startup from --openai-files-path.
void set_root(const std::string & root);
const std::string & root();

// Join root + relative; ensures trailing separator on root copy.
std::string join_dir(const std::string & root, const std::string & sub);

bool ensure_dir(const std::string & dir);

// Filename-safe id: keep [A-Za-z0-9_-] as-is; otherwise hex-encode as "h_<hex>"
// so distinct ids cannot collide (e.g. "a.b" vs "a_b").
std::string safe_id(const std::string & id);

// Pre-hex underscore-substitution encoding (for reading older on-disk files).
std::string safe_id_legacy(const std::string & id);

std::string json_path(const std::string & dir, const std::string & id);

bool write_json_file(const std::string & path, const json & obj);
bool read_json_file(const std::string & path, json & out);
void erase_file(const std::string & path);

// List *.json basenames (without path) under dir.
std::vector<std::string> list_json_basenames(const std::string & dir);

} // namespace openai_persist
