
#include <math.h>
#include <stdio.h>
#include "fastmath.h"
int main(void){ double el=0, ee=0;
  for (float x = 1.0f; x < 1.0e5f; x *= 1.0007f) { double d = fabs(fm_log10f(x) - log10f(x)); if (d > el) el = d; }
  for (float x = -20.0f; x < 20.0f; x += 0.01f) { double d = fabs(fm_expf(x) / expf(x) - 1.0); if (d > ee) ee = d; }
  printf("%g %g\n", el, ee); return 0; }
