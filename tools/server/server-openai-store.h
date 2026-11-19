#pragma once

#include "server-common.h"

#include <cstdint>
#include <string>
#include <vector>

// small helpers shared by the OpenAI protocol stores
namespace server_openai_store {

int64_t now();

std::string item_path(const std::string & base, const std::string & subdir, const std::string & id);

std::vector<std::string> list_files(const std::string & dir);

bool read_json(const std::string & path, json & data);
bool write_json(const std::string & path, const json & data);
bool remove_file(const std::string & path);

} // namespace server_openai_store
