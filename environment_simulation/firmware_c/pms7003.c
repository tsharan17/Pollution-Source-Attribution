#include "pms7003.h"

void pms_init(pms_parser_t *p)
{
    p->n = 0;
    p->frames_ok = 0;
    p->errors = 0;
}

int pms_feed_byte(pms_parser_t *p, uint8_t b, pms_frame_t *out)
{
    if (p->n == 0) {
        if (b == 0x42) p->buf[p->n++] = b;
        return 0;
    }
    if (p->n == 1) {
        if (b == 0x4D) p->buf[p->n++] = b;
        else if (b != 0x42) p->n = 0;   /* 0x42 0x42: stay synchronised on the new 0x42 */
        return 0;
    }
    p->buf[p->n++] = b;
    if (p->n == 4 && (((uint16_t)p->buf[2] << 8) | p->buf[3]) != 28) {
        p->errors++;                    /* bad frame length: resync */
        p->n = 0;
        return 0;
    }
    if (p->n == 32) {
        uint16_t cs = 0;
        for (int i = 0; i < 30; i++) cs += p->buf[i];
        p->n = 0;
        if (cs != (((uint16_t)p->buf[30] << 8) | p->buf[31])) {
            p->errors++;
            return 0;
        }
        uint16_t w[12];
        for (int k = 0; k < 12; k++)
            w[k] = ((uint16_t)p->buf[4 + 2 * k] << 8) | p->buf[5 + 2 * k];
        for (int k = 0; k < 3; k++) { out->cf1[k] = w[k]; out->atm[k] = w[3 + k]; }
        for (int k = 0; k < 6; k++) out->cum[k] = w[6 + k];
        p->frames_ok++;
        return 1;
    }
    return 0;
}
