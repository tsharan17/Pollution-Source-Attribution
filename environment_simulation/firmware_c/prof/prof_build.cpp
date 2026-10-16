// Operation-counting build of the PolSo firmware core.
//
// The unmodified firmware sources are compiled with `float` replaced by a counting class `pf`,
// so every add / multiply / divide / compare / conversion and every libm call executed on a real
// sensor stream is tallied per processing stage (UART parse, 1 Hz sample, window checks, features,
// MLP, post-processing). Counts are exact for the source as written; an ARM compiler may remove
// some operations (common sub-expressions), so they are an upper bound. Multiply by Cortex-M4F
// cycle costs to get time and energy (stm32_twin/profile.py).
//
// Input: the same PSIL binary stream as sil_main.c.  Usage: prof_exe in.bin out.csv > counts.json
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

enum { PH_NONE, PH_UART, PH_SAMPLE, PH_WCHECK, PH_FEAT, PH_MLP, PH_POST, PH_N };
enum { K_ADD, K_MUL, K_DIV, K_CMP, K_CVT, K_SQRT, K_LOG, K_LOG10, K_EXP, K_FLOOR, K_FM, K_N };
static const char *PH_NAME[PH_N] = {"none", "uart", "sample", "wcheck", "features", "mlp", "post"};
static const char *K_NAME[K_N] = {"add", "mul", "div", "cmp", "cvt", "sqrt", "log", "log10", "exp", "floor", "fmcall"};
static uint64_t g_ops[PH_N][K_N];
static int g_ph = 0;
static inline void cnt(int k) { g_ops[g_ph][k]++; }

struct pf {
    float v;
    pf() : v(0.0f) {}
    pf(float x) : v(x) {}
    pf(double x) : v((float)x) {}
    template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    pf(T x) : v((float)x) { cnt(K_CVT); }
    explicit operator float() const { return v; }
    template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    explicit operator T() const { cnt(K_CVT); return (T)v; }
    pf &operator+=(pf b) { v += b.v; cnt(K_ADD); return *this; }
    pf &operator-=(pf b) { v -= b.v; cnt(K_ADD); return *this; }
    pf &operator*=(pf b) { v *= b.v; cnt(K_MUL); return *this; }
    pf &operator/=(pf b) { v /= b.v; cnt(K_DIV); return *this; }
};
static inline pf operator+(pf a, pf b) { cnt(K_ADD); return pf(a.v + b.v); }
static inline pf operator-(pf a, pf b) { cnt(K_ADD); return pf(a.v - b.v); }
static inline pf operator*(pf a, pf b) { cnt(K_MUL); return pf(a.v * b.v); }
static inline pf operator/(pf a, pf b) { cnt(K_DIV); return pf(a.v / b.v); }
static inline pf operator-(pf a) { cnt(K_ADD); return pf(-a.v); }
static inline bool operator<(pf a, pf b) { cnt(K_CMP); return a.v < b.v; }
static inline bool operator>(pf a, pf b) { cnt(K_CMP); return a.v > b.v; }
static inline bool operator<=(pf a, pf b) { cnt(K_CMP); return a.v <= b.v; }
static inline bool operator>=(pf a, pf b) { cnt(K_CMP); return a.v >= b.v; }
static inline bool operator==(pf a, pf b) { cnt(K_CMP); return a.v == b.v; }
static inline bool operator!=(pf a, pf b) { cnt(K_CMP); return a.v != b.v; }

static inline pf prof_logf(pf x) { cnt(K_LOG); return pf(::logf(x.v)); }
static inline pf prof_log10f(pf x) { cnt(K_LOG10); return pf(::log10f(x.v)); }
static inline pf prof_expf(pf x) { cnt(K_EXP); return pf(::expf(x.v)); }
static inline pf prof_sqrtf(pf x) { cnt(K_SQRT); return pf(::sqrtf(x.v)); }
static inline pf prof_floorf(pf x) { cnt(K_FLOOR); return pf(::floorf(x.v)); }

#define float pf
#define logf prof_logf
#define log10f prof_log10f
#define expf prof_expf
#define sqrtf prof_sqrtf
#define floorf prof_floorf
#define PROF_PHASE(p) (g_ph = (p))
#define PROF_FM_CALL() cnt(K_FM)

#include "../pms7003.c"
#include "../polso_fw.c"

#undef float
#undef logf
#undef log10f
#undef expf
#undef sqrtf
#undef floorf

static int rd(void *p, size_t n, FILE *f) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: prof in.bin out.csv\n"); return 2; }
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "w");
    if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 2; }
    char magic[4];
    uint32_t n;
    float b0f[FW_N_STEPS];
    if (!rd(magic, 4, in) || memcmp(magic, "PSIL", 4) || !rd(&n, 4, in) || !rd(b0f, sizeof b0f, in)) return 2;
    pf b0[FW_N_STEPS];
    for (int i = 0; i < FW_N_STEPS; i++) b0[i] = pf(b0f[i]);
    static fw_state_t st;
    fw_result_t res;
    fw_init(&st, b0);
    fprintf(out, "t_end,label");
    for (int c = 0; c < FW_N_CLASSES; c++) fprintf(out, ",p%d", c);
    for (int k = 0; k < FW_N_FEAT; k++) fprintf(out, ",f%d", k);
    fprintf(out, "\n");
    uint8_t buf[512];
    uint64_t n_windows = 0, n_feat = 0, n_frames_bytes = 0;
    for (uint32_t s = 0; s < n; s++) {
        uint16_t nb; float r, t, rh; uint8_t step, ok;
        if (!rd(&nb, 2, in) || nb > sizeof buf || !rd(buf, nb, in) || !rd(&r, 4, in) || !rd(&step, 1, in) ||
            !rd(&ok, 1, in) || !rd(&t, 4, in) || !rd(&rh, 4, in)) { fprintf(stderr, "truncated\n"); return 3; }
        n_frames_bytes += nb;
        fw_uart_rx(&st, buf, nb);
        if (fw_tick(&st, pf(r), step, ok, pf(t), pf(rh), &res)) {
            n_windows++;
            if (res.label != FW_LBL_WARMUP && res.label != FW_LBL_FAULT) n_feat++;
            fprintf(out, "%u,%d", res.t_end, (int)res.label);
            for (int c = 0; c < FW_N_CLASSES; c++) fprintf(out, ",%.7g", res.probs[c].v);
            for (int k = 0; k < FW_N_FEAT; k++) fprintf(out, ",%.7g", res.feat[k].v);
            fprintf(out, "\n");
        }
    }
    fclose(out);
    printf("{\"n_ticks\":%u,\"n_windows\":%llu,\"n_feature_windows\":%llu,\"uart_bytes\":%llu,\"frames_ok\":%u,\"state_bytes\":%u,\"ops\":{",
           n, (unsigned long long)n_windows, (unsigned long long)n_feat, (unsigned long long)n_frames_bytes,
           st.pms.frames_ok, (unsigned)sizeof(fw_state_t));
    for (int p = 1; p < PH_N; p++) {
        printf("%s\"%s\":{", p > 1 ? "," : "", PH_NAME[p]);
        for (int k = 0; k < K_N; k++) printf("%s\"%s\":%llu", k ? "," : "", K_NAME[k], (unsigned long long)g_ops[p][k]);
        printf("}");
    }
    printf("}}\n");
    return 0;
}
