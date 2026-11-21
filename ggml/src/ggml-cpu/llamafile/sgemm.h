#pragma once
#include <stdint.h>
#include <stdbool.h>


#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);

#ifdef __cplusplus
}
#endif
