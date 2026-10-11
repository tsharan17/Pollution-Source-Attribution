/* PMS7003 active-mode frame parser (32-byte frames, 0x42 0x4D header, sum checksum).
 * Byte-at-a-time state machine: safe to call from a UART RX / DMA half-transfer callback. */
#ifndef PMS7003_H
#define PMS7003_H
#include <stdint.h>

typedef struct {
    uint16_t cf1[3];   /* PM1.0, PM2.5, PM10 (CF=1), ug/m3 */
    uint16_t atm[3];   /* same, atmospheric environment */
    uint16_t cum[6];   /* particles > 0.3/0.5/1.0/2.5/5/10 um per 0.1 L */
} pms_frame_t;

typedef struct {
    uint8_t buf[32];
    uint8_t n;
    uint32_t frames_ok;
    uint32_t errors;
} pms_parser_t;

void pms_init(pms_parser_t *p);
/* Returns 1 when a complete, checksum-valid frame has been written to *out. */
int pms_feed_byte(pms_parser_t *p, uint8_t b, pms_frame_t *out);

#endif
