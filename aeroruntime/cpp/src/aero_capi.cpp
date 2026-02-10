/**
 * AERO C API — thin wrapper around C++ AeroFile.
 * Exceptions are caught and converted to error codes; last error via aero_error_string().
 */
#include "aero/aero_capi.h"
#include "aero/aero.hpp"
#include "aero/aero_internal.h"

#include <cstring>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

void set_error(const char* msg) {
    g_last_error = msg ? msg : "";
}

} // namespace

extern "C" {

void aero_internal_set_error(const char* msg) {
    g_last_error = msg ? msg : "";
}


aero_handle_t aero_load(const char* path) {
    if (!path) {
        set_error("path is NULL");
        return nullptr;
    }
    try {
        auto* f = new aero::AeroFile(path);
        return reinterpret_cast<aero_handle_t>(f);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown error");
        return nullptr;
    }
}

void aero_close(aero_handle_t h) {
    if (!h) return;
    try {
        delete reinterpret_cast<aero::AeroFile*>(h);
    } catch (...) {
        set_error("error during close");
    }
}

uint32_t aero_tensor_count(aero_handle_t h) {
    if (!h) {
        set_error("handle is NULL");
        return 0;
    }
    try {
        auto* f = reinterpret_cast<aero::AeroFile*>(h);
        auto tensors = f->load_tidx();
        return static_cast<uint32_t>(tensors.size());
    } catch (const std::exception& e) {
        set_error(e.what());
        return 0;
    } catch (...) {
        set_error("unknown error");
        return 0;
    }
}

int aero_tensor_name(aero_handle_t h, uint32_t i, char* buffer, size_t buffer_size) {
    if (!h || !buffer) {
        set_error("handle or buffer is NULL");
        return -1;
    }
    if (buffer_size == 0) return -1;
    try {
        auto* f = reinterpret_cast<aero::AeroFile*>(h);
        auto tensors = f->load_tidx();
        if (i >= tensors.size()) {
            set_error("tensor index out of range");
            return -1;
        }
        const std::string& name = tensors[i].name;
        size_t len = name.size();
        if (len >= buffer_size) {
            set_error("buffer too small");
            return -1;
        }
        std::memcpy(buffer, name.data(), len);
        buffer[len] = '\0';
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown error");
        return -1;
    }
}

int aero_tensor_info(aero_handle_t h, const char* name,
                     aero_dtype_t* dtype_out, int64_t* shape_out, uint32_t* rank_out,
                     uint64_t* data_len_out) {
    if (!h || !name) {
        set_error("handle or name is NULL");
        return -1;
    }
    try {
        auto* f = reinterpret_cast<aero::AeroFile*>(h);
        auto tensors = f->load_tidx();
        const aero::TensorEntry* te = nullptr;
        for (const auto& t : tensors) {
            if (t.name == name) {
                te = &t;
                break;
            }
        }
        if (!te) {
            set_error("tensor not found");
            return -1;
        }
        if (dtype_out) *dtype_out = static_cast<aero_dtype_t>(te->dtype);
        if (data_len_out) *data_len_out = te->data_len;
        uint32_t rank = static_cast<uint32_t>(te->shape.size());
        if (rank_out) *rank_out = rank;
        if (shape_out) {
            size_t copy = rank <= 8 ? rank : 8;
            for (size_t i = 0; i < copy; ++i)
                shape_out[i] = te->shape[i];
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown error");
        return -1;
    }
}

int aero_tensor_ptr(aero_handle_t h, const char* name,
                    const uint8_t** data_out, uint64_t* size_out) {
    if (!h || !name || !data_out || !size_out) {
        set_error("NULL argument");
        return -1;
    }
    try {
        auto* f = reinterpret_cast<aero::AeroFile*>(h);
        auto view = f->view_tensor(name);
        *data_out = view.data;
        *size_out = view.size;
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown error");
        return -1;
    }
}

int aero_tensor_bytes(aero_handle_t h, const char* name,
                      uint8_t* buffer, uint64_t buffer_size) {
    if (!h || !name || !buffer) {
        set_error("NULL argument");
        return -1;
    }
    try {
        auto* f = reinterpret_cast<aero::AeroFile*>(h);
        std::vector<uint8_t> data = f->read_tensor_bytes(name);
        if (buffer_size < data.size()) {
            set_error("buffer too small");
            return -1;
        }
        std::memcpy(buffer, data.data(), data.size());
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown error");
        return -1;
    }
}

const char* aero_error_string(void) {
    static const char empty[] = "";
    return g_last_error.empty() ? empty : g_last_error.c_str();
}

} // extern "C"
