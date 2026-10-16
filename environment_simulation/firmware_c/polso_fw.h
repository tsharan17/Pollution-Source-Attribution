/* PolSo edge firmware core: sampling store, gas baseline, features, int8 MLP.
 * Portable C99, no dynamic memory. Hardware glue (HAL UART DMA, I2C, SD, GPS) calls:
 *     fw_uart_rx()  from the PMS7003 UART receive callback
 *     fw_tick()     once per second after reading the BME688 heater step            */
#ifndef POLSO_FW_H
#define POLSO_FW_H
#include <stdint.h>
#include "pms7003.h"
#include "model_params.h"

#define FW_N_FEAT 25
#define FW_N_CLASSES 5

typedef enum {
    FW_LBL_VEHICLE = 0, FW_LBL_BIOMASS, FW_LBL_DUST, FW_LBL_INDUSTRIAL, FW_LBL_BACKGROUND,
    FW_LBL_UNCERTAIN, FW_LBL_FAULT, FW_LBL_WARMUP, FW_LBL_MIXED
} fw_label_t;

#define REP_PM_N 10          /* 5-minute PM2.5 average (windows) */
#define REP_LAB_N 20         /* 10-minute dominant source (windows) */
#define REP_MIN_ANSWERS 6

typedef struct {
    /* 1 Hz ring buffers, slot = second % FW_WINDOW_S */
    float cum[FW_WINDOW_S][6];
    float pm[FW_WINDOW_S][3];
    uint8_t pm_ok[FW_WINDOW_S];
    float r[FW_WINDOW_S];
    uint8_t step[FW_WINDOW_S];
    uint8_t gas_ok[FW_WINDOW_S];
    float t_meas[FW_WINDOW_S];
    float rh_meas[FW_WINDOW_S];
    float base[FW_N_STEPS];          /* clean-air ln R per heater step */
    float probs_prev[FW_N_CLASSES];
    uint8_t prev_ok;
    float ah_cache;                  /* absolute humidity used by the gas compensation */
#ifdef FW_OPT_AH_DECIMATE
    float t_acc, rh_acc;             /* 10-sample accumulators for the humidity term */
    uint8_t n_acc, ah_set;
#endif
#ifdef FW_OPT_LOGCACHE
    float lc0[FW_WINDOW_S];          /* log10(1 + count>0.3um) per second */
#endif
    /* reporting layer history */
    float rep_pm[REP_PM_N], rep_lo[REP_PM_N], rep_hi[REP_PM_N];
    uint8_t rep_pm_ok[REP_PM_N];
    uint8_t rep_lab[REP_LAB_N];
    uint32_t rep_n;
    uint32_t sec;                    /* index of the next sample since power-on */
    pms_parser_t pms;
    pms_frame_t last_frame;
    uint8_t frame_this_sec;
} fw_state_t;

typedef struct {
    uint32_t t_end;
    fw_label_t label;
    float feat[FW_N_FEAT];
    float probs[FW_N_CLASSES];
    float probs_raw[FW_N_CLASSES];
    float pm_raw;                    /* window mean PM2.5 (CF=1) from the PMS7003 */
    float pm_corr, pm_lo, pm_hi;     /* corrected PM2.5 and its 16-84 % band */
    float pm5, pm5_lo, pm5_hi;       /* 5-minute average of the corrected value */
    fw_label_t label10;              /* 10-minute dominant source */
} fw_result_t;

void fw_init(fw_state_t *s, const float b0[FW_N_STEPS]);
void fw_uart_rx(fw_state_t *s, const uint8_t *data, uint32_t n);
int fw_tick(fw_state_t *s, float gas_r, uint8_t step, uint8_t gas_ok,
            float t_meas, float rh_meas, fw_result_t *out);
void fw_mlp_forward(const float x[FW_N_FEAT], float probs[FW_N_CLASSES]);
uint32_t fw_mlp_macs(void);

#endif
