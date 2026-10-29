#include "dosling_scaler.h"
#include "student-v37.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { W = 8, H = 6, SS = W * 4 + 8, DS = W * 12 + 16 };
    uint8_t source[SS * H], output[DS * H * 3];
    memset(source, 0xA5, sizeof source);
    memset(output, 0xCD, sizeof output);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t *p = source + y * SS + x * 4;
            p[0] = 17; p[1] = 91; p[2] = 233; p[3] = 0;
        }
    }
    dl_context *c = dl_create_from_memory(
        dos_lut_student_v37, dos_lut_student_v37_size);
    if (!c) return 10;
    if (dl_set_workers(c, 1) != 0) { dl_destroy(c); return 11; }
    dl_stats stats = {0};
    int status = dl_scale3_bgra(
        c, source, W, H, SS, output, DS, NULL, 0, &stats);
    int bad = status != 0 || stats.pixels != W * H ||
              dl_model_bytes(c) != 30238;
    if (status == 0) {
        for (int y = 0; y < H * 3; ++y) {
            for (int x = 0; x < W * 3; ++x) {
                const uint8_t *p = output + y * DS + x * 4;
                bad |= p[0] != 17 || p[1] != 91 ||
                       p[2] != 233 || p[3] != 255;
            }
            for (int b = W * 12; b < DS; ++b)
                bad |= output[y * DS + b] != 0xCD;
        }
    }
    dl_destroy(c);
    if (bad) return 12;
    puts("V42 API smoke passed");
    return 0;
}
