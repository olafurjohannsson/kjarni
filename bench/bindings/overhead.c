// The baseline: the C ABI called directly, as every other binding ends up doing.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kjarni.h"

#define WARMUP 200
#define CALLS 3000

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

int main(void) {
    struct KjarniEmbedderConfig config = kjarni_embedder_config_default();
    config.model_name = "minilm-l6-v2";
    config.quiet = 1;
    struct KjarniEmbedder *embedder = NULL;
    if (kjarni_embedder_new(&config, &embedder) != KJARNI_ERROR_CODE_OK) return 1;

    const char *text = "how do I get a refund for a damaged order?";
    static double times[CALLS];
    for (int i = 0; i < WARMUP + CALLS; i++) {
        struct KjarniFloatArray out;
        double start = now_us();
        if (kjarni_embedder_encode(embedder, text, &out) != KJARNI_ERROR_CODE_OK) return 1;
        kjarni_float_array_free(out);
        if (i >= WARMUP) times[i - WARMUP] = now_us() - start;
    }
    qsort(times, CALLS, sizeof(double), cmp);
    printf("{\"lang\": \"C\", \"median_us\": %.1f, \"min_us\": %.1f}\n", times[CALLS / 2], times[0]);
    return 0;
}
