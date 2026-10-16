/* Fast log / exp for the Cortex-M4F: IEEE-754 exponent extraction plus a short polynomial.
 * No libm call, no table, one floor adjustment in exp.
 *   fm_log2f   max abs error 6e-5   (features use log10 of counts ~1e2..1e5, so < 2e-5 absolute)
 *   fm_exp2f   max rel error 1e-7 (polynomial); fm_expf ~1e-6 for |x| <= 20 (float rounding of x*log2(e))
 * Bit casts use memcpy (compiles to a register move) so the same header also builds under the
 * operation-counting harness in prof/. Enabled with -DFW_OPT_FASTMATH.                            */
#ifndef FASTMATH_H
#define FASTMATH_H
#include <stdint.h>
#include <string.h>

#ifndef PROF_FM_CALL
#define PROF_FM_CALL() ((void)0)   /* counted by the profiling harness */
#endif

static inline float fm_log2f(float x)
{
    PROF_FM_CALL();
    uint32_t i;
    memcpy(&i, &x, 4);
    int32_t e = (int32_t)((i >> 23) & 0xFFu) - 127;
    i = (i & 0x007FFFFFu) | 0x3F800000u;
    float m;
    memcpy(&m, &i, 4);                         /* mantissa in [1,2) */
    float t = m - 1.0f;
    float q = (((5.994049644e-02f * t - 2.277011899e-01f) * t + 4.422655886e-01f) * t
               - 7.170615458e-01f) * t + 1.442615504e+00f;
    return (float)e + t * q;
}

static inline float fm_exp2f(float x)
{
    PROF_FM_CALL();
    if (x < -126.0f) x = -126.0f;
    if (x > 126.0f) x = 126.0f;
    int32_t n = (int32_t)x;                    /* truncates toward zero */
    if (x < (float)n) n--;                     /* floor */
    float f = x - (float)n;                    /* [0,1) */
    float p = ((((1.895107291e-03f * f + 8.946214667e-03f) * f + 5.586328266e-02f) * f
                + 2.401407701e-01f) * f + 6.931546200e-01f) * f + 9.999998958e-01f;
    uint32_t b = (uint32_t)(n + 127) << 23;
    float s;
    memcpy(&s, &b, 4);
    return p * s;
}

static inline float fm_logf(float x)   { return fm_log2f(x) * 0.69314718f; }
static inline float fm_log10f(float x) { return fm_log2f(x) * 0.30102999f; }
static inline float fm_expf(float x)   { return fm_exp2f(x * 1.44269504f); }

#endif
