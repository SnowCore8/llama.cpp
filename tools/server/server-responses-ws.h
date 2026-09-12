#pragma once

#include <functional>
#include <map>
#include <string>

#include "server-http.h"

// Runs one Responses WebSocket session: reads client events, runs response.create
// lanes, and writes server events. poll_text returns 1 text / 0 timeout / -1 closed.
void server_responses_ws_run(
        const std::function<server_http_res_ptr(const server_http_req &)> & create_fn,
        const std::map<std::string, std::string> & headers,
        const std::function<int(std::string &, int)> & poll_text,
        const std::function<bool(const std::string &)> & send_text);
