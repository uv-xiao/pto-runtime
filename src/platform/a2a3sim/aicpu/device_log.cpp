/**
 * @file device_log.cpp
 * @brief Simulation Platform Log Implementation
 *
 * Provides log enable flags and initialization for simulation environment.
 * Log levels can be controlled via PTO_LOG_LEVEL environment variable.
 */

#include "aicpu/device_log.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

// =============================================================================
// Log Enable Flags (Simulation: controlled by PTO_LOG_LEVEL)
// =============================================================================

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = true;
bool g_is_log_enable_warn = true;
bool g_is_log_enable_error = true;

// =============================================================================
// Platform Constant
// =============================================================================

const char* TILE_FWK_DEVICE_MACHINE = "SIM_CPU";

// Optional log file (shares PTO_LOG_FILE with host logger).
// In simulation, we *tee* logs to both stdout and the file (if configured)
// so profiling runs remain visible while still being parsable offline.
static FILE* g_device_log_file = nullptr;

static FILE* get_device_log_file() {
    if (g_device_log_file != nullptr) {
        return g_device_log_file;
    }

    const char* file_path = std::getenv("PTO_LOG_FILE");
    if (file_path == nullptr || file_path[0] == '\0') {
        return nullptr;
    }

    g_device_log_file = std::fopen(file_path, "a");
    if (g_device_log_file != nullptr) {
        // Line-buffered to keep lines intact for parsing.
        std::setvbuf(g_device_log_file, nullptr, _IOLBF, 0);
    }
    return g_device_log_file;
}

// =============================================================================
// Log Initialization (Read from PTO_LOG_LEVEL environment variable)
// =============================================================================

void init_log_switch() {
    // Read PTO_LOG_LEVEL environment variable
    const char* level_str = std::getenv("PTO_LOG_LEVEL");
    
    if (level_str != nullptr) {
        // Convert to lowercase for comparison
        char level[64];
        strncpy(level, level_str, sizeof(level) - 1);
        level[sizeof(level) - 1] = '\0';
        
        for (char* p = level; *p; ++p) {
            *p = tolower(*p);
        }
        
        // Parse log level
        if (strcmp(level, "silent") == 0 || strcmp(level, "error") == 0) {
            // Silent/Error: only errors
            g_is_log_enable_debug = false;
            g_is_log_enable_info = false;
            g_is_log_enable_warn = false;
            g_is_log_enable_error = true;
        } else if (strcmp(level, "normal") == 0 || strcmp(level, "info") == 0) {
            // Normal/Info: info, warn, error (default)
            g_is_log_enable_debug = false;
            g_is_log_enable_info = true;
            g_is_log_enable_warn = true;
            g_is_log_enable_error = true;
        } else if (strcmp(level, "verbose") == 0 || strcmp(level, "debug") == 0) {
            // Verbose/Debug: all levels
            g_is_log_enable_debug = true;
            g_is_log_enable_info = true;
            g_is_log_enable_warn = true;
            g_is_log_enable_error = true;
        } else {
            // Unknown value: default to INFO
            g_is_log_enable_debug = false;
            g_is_log_enable_info = true;
            g_is_log_enable_warn = true;
            g_is_log_enable_error = true;
        }
    } else {
        // No environment variable: default to INFO level
        g_is_log_enable_debug = false;
        g_is_log_enable_info = true;
        g_is_log_enable_warn = true;
        g_is_log_enable_error = true;
    }
}

// =============================================================================
// Platform-Specific Logging Functions (Simulation: use printf)
// =============================================================================

static void vdev_log(FILE* fp, const char* level, const char* func, const char* fmt, va_list args) {
    ::flockfile(fp);
    std::fprintf(fp, "[%s] %s: ", level, func);
    std::vfprintf(fp, fmt, args);
    std::fprintf(fp, "\n");
    ::funlockfile(fp);
}

static void dev_log_tee(const char* level, const char* func, const char* fmt, va_list args) {
    // Always print to stdout for interactive visibility.
    va_list args_stdout;
    va_copy(args_stdout, args);
    vdev_log(stdout, level, func, fmt, args_stdout);
    va_end(args_stdout);

    // Optionally append to PTO_LOG_FILE.
    FILE* fp = get_device_log_file();
    if (fp != nullptr) {
        va_list args_file;
        va_copy(args_file, args);
        vdev_log(fp, level, func, fmt, args_file);
        va_end(args_file);
    }
}

void dev_log_debug(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    dev_log_tee("DEBUG", func, fmt, args);
    va_end(args);
}

void dev_log_info(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    dev_log_tee("INFO", func, fmt, args);
    va_end(args);
}

void dev_log_warn(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    dev_log_tee("WARN", func, fmt, args);
    va_end(args);
}

void dev_log_error(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    dev_log_tee("ERROR", func, fmt, args);
    va_end(args);
}

void dev_log_always(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    dev_log_tee("ALWAYS", func, fmt, args);
    va_end(args);
}
