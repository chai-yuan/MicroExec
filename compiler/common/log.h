#ifndef SIMPLE_LOG_H
#define SIMPLE_LOG_H

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef LOG_LEVEL
#define LOG_LEVEL 0
#endif

#define LOG_COLOR_RESET "\033[0m"
#define LOG_COLOR_CYAN "\033[36m"
#define LOG_COLOR_GREEN "\033[32m"
#define LOG_COLOR_YELLOW "\033[33m"
#define LOG_COLOR_RED "\033[31m"

static inline void _log_print(int level, const char *level_str, const char *color, const char *file, int line,
                              const char *fmt, ...) {
    FILE *stream = (level >= 2) ? stderr : stdout;

    time_t     t       = time(NULL);
    struct tm *tm_info = localtime(&t);
    char       time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

    fprintf(stream, "%s[%s] [%-5s] [%s:%d]: ", color, time_buf, level_str, file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stream, fmt, args);
    va_end(args);

    fprintf(stream, "%s\n", LOG_COLOR_RESET);
}

#if LOG_LEVEL <= 0
#define LOG_DEBUG(...) _log_print(0, "DEBUG", LOG_COLOR_CYAN, __FILE__, __LINE__, __VA_ARGS__)
#else
#define LOG_DEBUG(...)                                                                                                 \
    do {                                                                                                               \
    } while (0)
#endif

#if LOG_LEVEL <= 1
#define LOG_INFO(...) _log_print(1, "INFO", LOG_COLOR_GREEN, __FILE__, __LINE__, __VA_ARGS__)
#else
#define LOG_INFO(...)                                                                                                  \
    do {                                                                                                               \
    } while (0)
#endif

#if LOG_LEVEL <= 2
#define LOG_WARN(...) _log_print(2, "WARN", LOG_COLOR_YELLOW, __FILE__, __LINE__, __VA_ARGS__)
#else
#define LOG_WARN(...)                                                                                                  \
    do {                                                                                                               \
    } while (0)
#endif

#if LOG_LEVEL <= 3
#define LOG_ERROR(...) _log_print(3, "ERROR", LOG_COLOR_RED, __FILE__, __LINE__, __VA_ARGS__)
#else
#define LOG_ERROR(...)                                                                                                 \
    do {                                                                                                               \
    } while (0)
#endif

#endif // SIMPLE_LOG_Hs
