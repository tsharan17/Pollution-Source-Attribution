/* Software-in-the-loop harness: replays a recorded/simulated sensor stream through the
 * exact firmware core and writes one CSV row per window.
 *
 * Input (little-endian): "PSIL", uint32 n_seconds, float b0[FW_N_STEPS], then per second:
 *   uint16 n_uart_bytes, uint8 bytes[n], float gas_r, uint8 step, uint8 gas_ok,
 *   float t_meas, float rh_meas
 * Usage: sil <input.bin> <output.csv>                                                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "polso_fw.h"

static int rd(void *p, size_t n, FILE *f) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: sil in.bin out.csv\n"); return 2; }
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "w");
    if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 2; }
    char magic[4];
    uint32_t n;
    float b0[FW_N_STEPS];
    if (!rd(magic, 4, in) || memcmp(magic, "PSIL", 4) || !rd(&n, 4, in) || !rd(b0, sizeof b0, in)) {
        fprintf(stderr, "bad header\n"); return 2;
    }
    static fw_state_t st;
    fw_result_t res;
    fw_init(&st, b0);
    fprintf(out, "t_end,label");
    for (int c = 0; c < FW_N_CLASSES; c++) fprintf(out, ",p%d", c);
    for (int k = 0; k < FW_N_FEAT; k++) fprintf(out, ",f%d", k);
    fprintf(out, ",pm_raw,pm_corr,pm5,label10\n");
    uint8_t buf[512];
    for (uint32_t s = 0; s < n; s++) {
        uint16_t nb; float r, t, rh; uint8_t step, ok;
        if (!rd(&nb, 2, in) || nb > sizeof buf || !rd(buf, nb, in) || !rd(&r, 4, in) ||
            !rd(&step, 1, in) || !rd(&ok, 1, in) || !rd(&t, 4, in) || !rd(&rh, 4, in)) {
            fprintf(stderr, "truncated at second %u\n", s); return 3;
        }
        fw_uart_rx(&st, buf, nb);
        if (fw_tick(&st, r, step, ok, t, rh, &res)) {
            fprintf(out, "%u,%d", res.t_end, (int)res.label);
            for (int c = 0; c < FW_N_CLASSES; c++) fprintf(out, ",%.7g", res.probs[c]);
            for (int k = 0; k < FW_N_FEAT; k++) fprintf(out, ",%.7g", res.feat[k]);
            fprintf(out, ",%.7g,%.7g,%.7g,%d\n", res.pm_raw, res.pm_corr, res.pm5, (int)res.label10);
        }
    }
    fclose(out);
    fprintf(stderr, "frames_ok=%u frame_errors=%u sizeof(fw_state_t)=%u model_weight_bytes=%u macs=%u\n",
            st.pms.frames_ok, st.pms.errors, (unsigned)sizeof(fw_state_t),
            (unsigned)(sizeof(MP_W1) + sizeof(MP_B1) + sizeof(MP_W2) + sizeof(MP_B2) + sizeof(MP_W3) +
                       sizeof(MP_B3) + sizeof(MP_MU) + sizeof(MP_INV_SD)),
            fw_mlp_macs());
    return 0;
}
