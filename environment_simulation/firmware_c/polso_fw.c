#include <math.h>
#include <string.h>
#include "polso_fw.h"
#include "pm_cal_params.h"

/* Profiling hooks: no-ops unless the operation-counting harness (prof/prof_build.cpp) defines PROF_PHASE. */
#ifndef PROF_PHASE
#define PROF_PHASE(p) ((void)0)
#endif
#define FWPH_NONE 0
#define FWPH_UART 1
#define FWPH_SAMPLE 2
#define FWPH_WCHECK 3
#define FWPH_FEAT 4
#define FWPH_MLP 5
#define FWPH_POST 6

/* Optional computation optimisations (all off in the reference build):
 *   FW_OPT_LOGCACHE    log10 of the 0.3 um count is computed once per second and stored, not per use
 *   FW_OPT_AH_DECIMATE absolute humidity is recomputed every 10 s (T and RH move slowly)
 *   FW_OPT_FASTMATH    polynomial log/exp instead of libm                                           */
#ifdef FW_OPT_FASTMATH
#include "fastmath.h"
#define FW_LOG10F(x) fm_log10f(x)
#define FW_LOGF(x)   fm_logf(x)
#define FW_EXPF(x)   fm_expf(x)
#else
#define FW_LOG10F(x) log10f(x)
#define FW_LOGF(x)   logf(x)
#define FW_EXPF(x)   expf(x)
#endif
#ifdef FW_OPT_LOGCACHE
#define LC0(i) (s->lc0[i])
#else
#define LC0(i) FW_LOG10F(1.0f + s->cum[i][0])
#endif

void fw_init(fw_state_t *s, const float b0[FW_N_STEPS])
{
    memset(s, 0, sizeof(*s));
    for (int i = 0; i < FW_N_STEPS; i++) s->base[i] = b0[i];
    pms_init(&s->pms);
}

void fw_uart_rx(fw_state_t *s, const uint8_t *data, uint32_t n)
{
    pms_frame_t f;
    PROF_PHASE(FWPH_UART);
    for (uint32_t i = 0; i < n; i++)
        if (pms_feed_byte(&s->pms, data[i], &f)) { s->last_frame = f; s->frame_this_sec = 1; }
    PROF_PHASE(FWPH_NONE);
}

/* ------------------------------------------------------------------------- */
/* int8 MLP: integer hidden layers, float soft-max                            */
/* ------------------------------------------------------------------------- */
static int8_t requant(int32_t acc, int32_t m, int sh)
{
    if (acc < 0) acc = 0;                                   /* ReLU */
    int64_t t = ((int64_t)acc * m + ((int64_t)1 << (sh - 1))) >> sh;
    return (int8_t)(t > 127 ? 127 : t);
}

void fw_mlp_forward(const float x[FW_N_FEAT], float probs[FW_N_CLASSES])
{
    int8_t q0[MP_N_IN], q1[MP_H1], q2[MP_H2];
    for (int i = 0; i < MP_N_IN; i++) {
        float xs = (x[i] - MP_MU[i]) * MP_INV_SD[i];
        float v = xs * MP_INV_S0;
        v = (v >= 0.0f) ? floorf(v + 0.5f) : -floorf(-v + 0.5f);
        q0[i] = (int8_t)(v > 127.0f ? 127 : (v < -127.0f ? -127 : v));
    }
    for (int j = 0; j < MP_H1; j++) {
        int32_t acc = MP_B1[j];
        for (int i = 0; i < MP_N_IN; i++) acc += (int32_t)MP_W1[j * MP_N_IN + i] * q0[i];
        q1[j] = requant(acc, MP_L1_M, MP_L1_SH);
    }
    for (int j = 0; j < MP_H2; j++) {
        int32_t acc = MP_B2[j];
        for (int i = 0; i < MP_H1; i++) acc += (int32_t)MP_W2[j * MP_H1 + i] * q1[i];
        q2[j] = requant(acc, MP_L2_M, MP_L2_SH);
    }
    float logit[MP_N_OUT], mx = -1e30f, sum = 0.0f;
    for (int j = 0; j < MP_N_OUT; j++) {
        int32_t acc = MP_B3[j];
        for (int i = 0; i < MP_H2; i++) acc += (int32_t)MP_W3[j * MP_H2 + i] * q2[i];
        logit[j] = (float)acc * MP_L3_SCALE;
        if (logit[j] > mx) mx = logit[j];
    }
    for (int j = 0; j < MP_N_OUT; j++) { probs[j] = FW_EXPF(logit[j] - mx); sum += probs[j]; }
    for (int j = 0; j < MP_N_OUT; j++) probs[j] /= sum;
}

uint32_t fw_mlp_macs(void)
{
    return MP_N_IN * MP_H1 + MP_H1 * MP_H2 + MP_H2 * MP_N_OUT;
}

/* ------------------------------------------------------------------------- */
/* Window features (see polso_twin/firmware.py:window_features)              */
/* ------------------------------------------------------------------------- */
static void features(const fw_state_t *s, float *f)
{
    const int W = FW_WINDOW_S;
    int slot[FW_WINDOW_S];
    for (int k = 0; k < W; k++) slot[k] = (int)((s->sec + 1 + k) % W);   /* oldest .. newest */
    memset(f, 0, sizeof(float) * FW_N_FEAT);

    /* ---- PM ---- */
    float dsum[6] = {0}, c0sum = 0, pmsum[3] = {0};
    int npm = 0;
    for (int k = 0; k < W; k++) {
        int i = slot[k];
        if (!s->pm_ok[i]) continue;
        const float *c = s->cum[i];
        for (int b = 0; b < 5; b++) { float d = c[b] - c[b + 1]; dsum[b] += d > 0 ? d : 0; }
        dsum[5] += c[5];
        c0sum += c[0];
        for (int b = 0; b < 3; b++) pmsum[b] += s->pm[i][b];
        npm++;
    }
    float tot = 0;
    for (int b = 0; b < 6; b++) { dsum[b] /= npm; tot += dsum[b]; }
    for (int b = 0; b < 6; b++) f[b] = tot > 0 ? FW_LOG10F(dsum[b] / tot + 1e-4f) : -4.0f;
    float m0 = c0sum / npm;
    float pm1 = pmsum[0] / npm, pm25 = pmsum[1] / npm, pm10 = pmsum[2] / npm;
    f[6] = FW_LOG10F(1.0f + m0);
    f[7] = (pm25 + 1.0f) / (pm10 + 1.0f);
    f[8] = (pm1 + 1.0f) / (pm25 + 1.0f);
    float v2 = 0, v3 = 0;
    for (int k = 0; k < W; k++) {
        int i = slot[k];
        if (!s->pm_ok[i]) continue;
        float d = s->cum[i][0] - m0;
        v2 += d * d; v3 += d * d * d;
    }
    v2 /= npm; v3 /= npm;
    f[9] = sqrtf(v2) / (m0 + 1.0f);
    if (v2 > 1e-6f * (m0 + 1.0f) * (m0 + 1.0f)) {
        float sk = v3 / (v2 * sqrtf(v2));
        f[10] = sk > 10 ? 10 : (sk < -10 ? -10 : sk);
    }
    int have_rise = 0;
    float rise = 0;
    for (int k = 5; k < W; k++) {
        int i = slot[k], j = slot[k - 5];
        if (!s->pm_ok[i] || !s->pm_ok[j]) continue;
        float d = LC0(i) - LC0(j);
        if (!have_rise || d > rise) { rise = d; have_rise = 1; }
    }
    f[11] = have_rise ? rise : 0.0f;

    /* ---- gas ---- */
    float rg[FW_N_GROUPS] = {0}, rhs = 0, ts = 0;
    int ng[FW_N_GROUPS] = {0}, ngas = 0;
    for (int k = 0; k < W; k++) {
        int i = slot[k];
        if (!s->gas_ok[i]) continue;
        int g = FW_GROUP_OF_STEP[s->step[i]];
        rg[g] += s->r[i]; ng[g]++;
        rhs += s->rh_meas[i]; ts += s->t_meas[i]; ngas++;
    }
    float M = 0;
    for (int g = 0; g < FW_N_GROUPS; g++) { rg[g] /= ng[g]; M += rg[g]; }
    M /= FW_N_GROUPS;
    f[12] = M;
    for (int g = 0; g < FW_N_GROUPS; g++) f[13 + g] = rg[g] - M;
    const int slope_grp[2] = {3, 1}, slope_col[2] = {19, 20};   /* 300 C, 200 C */
    for (int q = 0; q < 2; q++) {
        float xm = 0, ym = 0; int nn = 0;
        for (int k = 0; k < W; k++) {
            int i = slot[k];
            if (!s->gas_ok[i] || FW_GROUP_OF_STEP[s->step[i]] != slope_grp[q]) continue;
            xm += k / 60.0f; ym += s->r[i]; nn++;
        }
        if (nn < 2) continue;
        xm /= nn; ym /= nn;
        float sxy = 0, sxx = 0;
        for (int k = 0; k < W; k++) {
            int i = slot[k];
            if (!s->gas_ok[i] || FW_GROUP_OF_STEP[s->step[i]] != slope_grp[q]) continue;
            float dx = k / 60.0f - xm;
            sxx += dx * dx; sxy += dx * (s->r[i] - ym);
        }
        f[slope_col[q]] = sxx > 0 ? sxy / sxx : 0.0f;
    }
    {   /* PM-gas co-variation at the 300 C step */
        float xm = 0, ym = 0; int nn = 0;
        for (int k = 0; k < W; k++) {
            int i = slot[k];
            if (!s->gas_ok[i] || FW_GROUP_OF_STEP[s->step[i]] != 3 || !s->pm_ok[i]) continue;
            xm += LC0(i); ym += -s->r[i]; nn++;
        }
        if (nn >= 3) {
            xm /= nn; ym /= nn;
            float sxx = 0, syy = 0, sxy = 0;
            for (int k = 0; k < W; k++) {
                int i = slot[k];
                if (!s->gas_ok[i] || FW_GROUP_OF_STEP[s->step[i]] != 3 || !s->pm_ok[i]) continue;
                float dx = LC0(i) - xm, dy = -s->r[i] - ym;
                sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
            }
            float sx = sqrtf(sxx / nn), sy = sqrtf(syy / nn);
            if (sx > 1e-6f && sy > 1e-6f) f[21] = (sxy / nn) / (sx * sy);
        }
    }
    f[22] = -rg[3] / (0.5f + FW_LOG10F(1.0f + pm25));
    f[23] = rhs / ngas / 100.0f;
    f[24] = ts / ngas / 50.0f;
}

static float ah_calc(float t_c, float rh)
{
    return 216.7f * (rh / 100.0f * 6.112f * FW_EXPF(17.62f * t_c / (243.12f + t_c))) / (273.15f + t_c);
}

/* ------------------------------------------------------------------------- */
/* Reporting layer: corrected PM2.5 (5-min average + band), 10-min dominant source */
/* (see polso_twin/firmware.py:report_layer)                                    */
/* ------------------------------------------------------------------------- */
static void fw_report(fw_state_t *s, fw_result_t *out, int valid)
{
    int ps = (int)(s->rep_n % REP_PM_N), ls = (int)(s->rep_n % REP_LAB_N);
    if (valid) {
        int k = out->label <= FW_LBL_BACKGROUND ? (int)out->label : 5;
        float rh = out->feat[23] * 100.0f;
        float a = (rh < 0.0f ? 0.0f : (rh > 95.0f ? 95.0f : rh)) / 100.0f;
        float corr = FW_EXPF(PMCAL[k][1] + PMCAL[k][0] * FW_LOGF(out->pm_raw + 1e-6f)) / (1.0f + PMCAL[k][2] * a / (1.0f - a));
        out->pm_corr = corr;
        out->pm_lo = corr * FW_EXPF(PMCAL[k][3]);
        out->pm_hi = corr * FW_EXPF(PMCAL[k][4]);
        s->rep_pm[ps] = out->pm_corr; s->rep_lo[ps] = out->pm_lo; s->rep_hi[ps] = out->pm_hi;
    }
    s->rep_pm_ok[ps] = (uint8_t)valid;
    s->rep_lab[ls] = valid ? (uint8_t)out->label : (uint8_t)FW_LBL_UNCERTAIN;
    s->rep_n++;
    int npm = s->rep_n < REP_PM_N ? (int)s->rep_n : REP_PM_N, c = 0;
    float sp = 0, sl = 0, sh = 0;
    for (int j = 0; j < npm; j++)
        if (s->rep_pm_ok[j]) { sp += s->rep_pm[j]; sl += s->rep_lo[j]; sh += s->rep_hi[j]; c++; }
    if (c) { out->pm5 = sp / c; out->pm5_lo = sl / c; out->pm5_hi = sh / c; }
    if (!valid) { out->label10 = out->label; return; }
    int nl = s->rep_n < REP_LAB_N ? (int)s->rep_n : REP_LAB_N, cnt[FW_N_CLASSES] = {0}, ans = 0;
    for (int j = 0; j < nl; j++)
        if (s->rep_lab[j] <= FW_LBL_BACKGROUND) { cnt[s->rep_lab[j]]++; ans++; }
    if (ans < REP_MIN_ANSWERS) { out->label10 = FW_LBL_UNCERTAIN; return; }
    static const int order[FW_N_CLASSES] = {FW_LBL_BACKGROUND, FW_LBL_BIOMASS, FW_LBL_DUST, FW_LBL_INDUSTRIAL, FW_LBL_VEHICLE};
    int best = order[0];
    for (int j = 1; j < FW_N_CLASSES; j++) if (cnt[order[j]] > cnt[best]) best = order[j];   /* ties: alphabetical, as numpy */
    out->label10 = (2 * cnt[best] >= ans) ? (fw_label_t)best : FW_LBL_MIXED;
}

/* ------------------------------------------------------------------------- */
/* 1 Hz tick                                                                  */
/* ------------------------------------------------------------------------- */
int fw_tick(fw_state_t *s, float gas_r, uint8_t step, uint8_t gas_ok,
            float t_meas, float rh_meas, fw_result_t *out)
{
    const int W = FW_WINDOW_S;
    int i = (int)(s->sec % W);
    PROF_PHASE(FWPH_SAMPLE);

    /* PM sample: last valid frame received during this second */
    s->pm_ok[i] = s->frame_this_sec;
    if (s->frame_this_sec) {
        for (int b = 0; b < 6; b++) s->cum[i][b] = (float)s->last_frame.cum[b];
        for (int b = 0; b < 3; b++) s->pm[i][b] = (float)s->last_frame.cf1[b];
#ifdef FW_OPT_LOGCACHE
        s->lc0[i] = FW_LOG10F(1.0f + s->cum[i][0]);
#endif
    }
    s->frame_this_sec = 0;

    /* gas sample: compensate, compare to baseline, then update baseline */
    uint8_t ok = gas_ok && gas_r >= FW_R_MIN && gas_r <= FW_R_MAX && step < FW_N_STEPS;
    s->gas_ok[i] = ok;
    s->step[i] = step;
    s->t_meas[i] = t_meas;
    s->rh_meas[i] = rh_meas;
    if (ok) {
#ifdef FW_OPT_AH_DECIMATE
        /* average 10 readings (the T/RH sensor is noisy), then evaluate the exponential once */
        s->t_acc += t_meas; s->rh_acc += rh_meas; s->n_acc++;
        if (!s->ah_set) {
            s->ah_cache = ah_calc(t_meas, rh_meas); s->ah_set = 1; s->t_acc = 0; s->rh_acc = 0; s->n_acc = 0;
        } else if (s->n_acc >= 10) {
            s->ah_cache = ah_calc(s->t_acc / 10.0f, s->rh_acc / 10.0f); s->t_acc = 0; s->rh_acc = 0; s->n_acc = 0;
        }
#else
        s->ah_cache = ah_calc(t_meas, rh_meas);
#endif
        float ah = s->ah_cache;
        float l = FW_LOGF(gas_r) + FW_GAMMA_STEP[step] * (ah - FW_AH_REF)
                  + FW_AMB_TEMP_COEF * ((t_meas - FW_SELF_HEAT) - 25.0f);
        float b = s->base[step];
        s->r[i] = l - b;
        if (s->sec >= FW_WARMUP_S) s->base[step] = (l > b) ? b + FW_BASE_UP * (l - b) : b - FW_BASE_DOWN;
    }

    uint32_t e = s->sec;
    s->sec++;
    if (e + 1 < (uint32_t)W || (e + 1 - W) % FW_HOP_S != 0) { PROF_PHASE(FWPH_NONE); return 0; }

    /* ---- window boundary ---- */
    PROF_PHASE(FWPH_WCHECK);
    memset(out, 0, sizeof(*out));
    out->t_end = e;
    s->sec--;                                  /* features() indexes relative to sample e */
    if (e < FW_WARMUP_S) { out->label = FW_LBL_WARMUP; s->prev_ok = 0; s->sec++; fw_report(s, out, 0); PROF_PHASE(FWPH_NONE); return 1; }
    int npm = 0, ngas = 0, seen[FW_N_GROUPS] = {0}, nseen = 0;
    for (int k = 0; k < W; k++) {
        npm += s->pm_ok[k];
        if (s->gas_ok[k]) {
            ngas++;
            int g = FW_GROUP_OF_STEP[s->step[k]];
            if (!seen[g]) { seen[g] = 1; nseen++; }
        }
    }
    if ((float)npm / W < FW_PM_VALID_FRAC || (float)ngas / W < FW_GAS_VALID_FRAC || nseen < FW_N_GROUPS) {
        out->label = FW_LBL_FAULT; s->prev_ok = 0; s->sec++; fw_report(s, out, 0); PROF_PHASE(FWPH_NONE); return 1;
    }
    PROF_PHASE(FWPH_FEAT);
    features(s, out->feat);
    s->sec++;
    PROF_PHASE(FWPH_MLP);
    fw_mlp_forward(out->feat, out->probs_raw);
    PROF_PHASE(FWPH_POST);
    for (int c = 0; c < FW_N_CLASSES; c++)
        out->probs[c] = s->prev_ok ? FW_PROB_EMA * s->probs_prev[c] + (1.0f - FW_PROB_EMA) * out->probs_raw[c]
                                   : out->probs_raw[c];
    s->prev_ok = 1;
    int best = 0;
    for (int c = 0; c < FW_N_CLASSES; c++) { s->probs_prev[c] = out->probs[c]; if (out->probs[c] > out->probs[best]) best = c; }
    out->label = out->probs[best] >= MP_CONF_THR ? (fw_label_t)best : FW_LBL_UNCERTAIN;
    {   /* window mean PM2.5 over the valid frames */
        float sp = 0; int c = 0;
        for (int k = 0; k < W; k++) if (s->pm_ok[k]) { sp += s->pm[k][1]; c++; }
        out->pm_raw = c ? sp / c : 0.0f;
    }
    fw_report(s, out, 1);
    PROF_PHASE(FWPH_NONE);
    return 1;
}
