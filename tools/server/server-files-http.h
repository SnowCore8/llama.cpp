#pragma once

#include "server-files.h"
#include "server-http.h"

namespace server_files_http {

server_http_res_ptr create(server_files_store & store, const server_http_req & req);
server_http_res_ptr list(server_files_store & store, const server_http_req & req);
server_http_res_ptr get(server_files_store & store, const server_http_req & req);
server_http_res_ptr del(server_files_store & store, const server_http_req & req);
server_http_res_ptr content(server_files_store & store, const server_http_req & req);

} // namespace server_files_http