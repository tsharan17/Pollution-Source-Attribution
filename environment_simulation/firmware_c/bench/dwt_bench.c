/* On-board benchmark for the STM32G474 (and host smoke test).
 *
 * Measures, with the Cortex-M4 DWT cycle counter, every quantity the STM32 twin estimates
 * (stm32_twin/spec.py), so the estimates can be replaced by measurements:
 *     libm calls, PMS frame parse, 1 Hz sample task, window task (features + MLP), MLP alone.
 * Output is one line per item on the debug UART / semihosting stdout:
 *     BENCH <name> <min_cycles> <mean_cycles> <max_cycles>
 * Paste those lines into a file and run:  python -m stm32_twin.compare_measured <file>
 *
 * Use: add polso_fw.c, pms7003.c, bench/dwt_bench.c to the CubeIDE project (-O2 or -Os, hard-float,
 * the same flags as the release build), call polso_bench_run() once from main() after clock setup.
 * Host build (syntax check only):  gcc -DBENCH_HOST -I. bench/dwt_bench.c polso_fw.c pms7003.c -lm       */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../polso_fw.h"

#ifdef BENCH_HOST
#include <time.h>
static inline void cyc_init(void) {}
static inline uint32_t cyc_now(void) { return (uint32_t)clock(); }
#else
#include "stm32g4xx.h"
static inline void cyc_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t cyc_now(void) { return DWT->CYCCNT; }
#endif

#define REPS 200

typedef struct { uint32_t min, max; uint64_t sum; uint32_t n; } stat_t;
static void st_add(stat_t *s, uint32_t c)
{
    if (s->n == 0 || c < s->min) s->min = c;
    if (c > s->max) s->max = c;
    s->sum += c; s->n++;
}
static void st_print(const char *name, const stat_t *s, uint32_t overhead)
{
    uint32_t mn = s->min > overhead ? s->min - overhead : 0;
    uint32_t mx = s->max > overhead ? s->max - overhead : 0;
    uint32_t mean = (uint32_t)(s->sum / s->n);
    mean = mean > overhead ? mean - overhead : 0;
    printf("BENCH %s %lu %lu %lu\r\n", name, (unsigned long)mn, (unsigned long)mean, (unsigned long)mx);
}

static volatile float sink;

static uint32_t measure_overhead(void)
{
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 50; i++) {
        uint32_t t0 = cyc_now();
        uint32_t t1 = cyc_now();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

#define BENCH_FN(NAME, EXPR)                                                  \
    do {                                                                      \
        stat_t s = {0, 0, 0, 0};                                              \
        for (int r = 0; r < REPS; r++) {                                      \
            float x = 50.0f + 37.0f * (float)r;                               \
            uint32_t t0 = cyc_now();                                          \
            sink = (EXPR);                                                    \
            st_add(&s, cyc_now() - t0);                                       \
        }                                                                     \
        st_print(NAME, &s, ovh);                                              \
    } while (0)

/* A frame with plausible counts, built the way the sensor sends it. */
static void make_frame(uint8_t f[32], uint16_t base)
{
    uint16_t w[14] = {28, 12, 25, 31, 12, 25, 31, base, (uint16_t)(base / 3), (uint16_t)(base / 12),
                      (uint16_t)(base / 90), 4, 1, 0x9700};
    f[0] = 0x42; f[1] = 0x4D;
    for (int k = 0; k < 14; k++) { f[2 + 2 * k] = (uint8_t)(w[k] >> 8); f[3 + 2 * k] = (uint8_t)w[k]; }
    uint16_t cs = 0;
    for (int i = 0; i < 30; i++) cs += f[i];
    f[30] = (uint8_t)(cs >> 8); f[31] = (uint8_t)cs;
}

void polso_bench_run(void)
{
    cyc_init();
    uint32_t ovh = measure_overhead();
    printf("BENCH overhead %lu %lu %lu\r\n", (unsigned long)ovh, (unsigned long)ovh, (unsigned long)ovh);

    /* ---- libm ---- */
    BENCH_FN("logf", logf(x));
    BENCH_FN("log10f", log10f(x));
    BENCH_FN("expf", expf(x * 0.001f));
    BENCH_FN("sqrtf", sqrtf(x));
    BENCH_FN("floorf", floorf(x * 0.37f));

    /* ---- parser ---- */
    {
        static pms_parser_t p;
        pms_frame_t out;
        uint8_t fr[32];
        pms_init(&p);
        stat_t s = {0, 0, 0, 0};
        for (int r = 0; r < REPS; r++) {
            make_frame(fr, (uint16_t)(3000 + 17 * r));
            uint32_t t0 = cyc_now();
            for (int i = 0; i < 32; i++) pms_feed_byte(&p, fr[i], &out);
            st_add(&s, cyc_now() - t0);
        }
        st_print("pms_frame_parse", &s, ovh);
    }

    /* ---- full firmware: feed 5 minutes of synthetic data, time each stage ---- */
    {
        static fw_state_t st;
        static float b0[FW_N_STEPS];
        fw_result_t res;
        uint8_t fr[32];
        stat_t t_sample = {0, 0, 0, 0}, t_window = {0, 0, 0, 0}, t_uart = {0, 0, 0, 0};
        for (int i = 0; i < FW_N_STEPS; i++) b0[i] = 11.7f;
        fw_init(&st, b0);
        for (uint32_t sec = 0; sec < 420; sec++) {
            make_frame(fr, (uint16_t)(3000 + 40 * (sec % 17)));
            uint32_t t0 = cyc_now();
            fw_uart_rx(&st, fr, 32);
            st_add(&t_uart, cyc_now() - t0);
            uint8_t step = (uint8_t)(sec % FW_N_STEPS);
            float r_ohm = 90000.0f + 3000.0f * (float)step - 20.0f * (float)(sec % 11);
            t0 = cyc_now();
            int win = fw_tick(&st, r_ohm, step, 1, 28.0f, 55.0f, &res);
            uint32_t dt = cyc_now() - t0;
            if (win && res.label != FW_LBL_WARMUP && res.label != FW_LBL_FAULT) st_add(&t_window, dt);
            else if (!win) st_add(&t_sample, dt);
        }
        st_print("fw_uart_rx", &t_uart, ovh);
        st_print("fw_tick_sample", &t_sample, ovh);
        st_print("fw_tick_window", &t_window, ovh);   /* sample part + features + MLP + post */
        {
            float x[FW_N_FEAT], pr[FW_N_CLASSES];
            for (int k = 0; k < FW_N_FEAT; k++) x[k] = 0.1f * (float)k - 0.7f;
            stat_t s = {0, 0, 0, 0};
            for (int r = 0; r < REPS; r++) {
                x[3] += 0.001f;
                uint32_t t0 = cyc_now();
                fw_mlp_forward(x, pr);
                st_add(&s, cyc_now() - t0);
                sink = pr[0];
            }
            st_print("mlp_forward", &s, ovh);
        }
    }
    printf("BENCH done 0 0 0\r\n");
}

#ifdef BENCH_HOST
int main(void) { polso_bench_run(); return 0; }
#endif
