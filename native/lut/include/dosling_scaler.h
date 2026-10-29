/* Dosling scaler: public C API. The dl_* ABI remains compatible with V42. */
#ifndef DOSLING_SCALER_H
#define DOSLING_SCALER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct dl_context dl_context;
typedef struct dl_rect {
  int x0, y0, x1, y1;
} dl_rect;
typedef struct dl_stats {
  // shade_checker is a legacy ABI name: V36 counts ordinary-plane bank2 selections.
  // text counts caller-hinted source pixels; general AA does not imply OCR.
  uint64_t pixels, extended, hq_fallback, text, shade_checker;
} dl_stats;
/* Optional diagnostics for the last successful render. Set struct_size. */
typedef struct dl_detail_stats {
  uint32_t struct_size;
  uint64_t ufo_corners, ufo_joins, ufo_wedges, exception_hits, exception_misses;
} dl_detail_stats;
int dl_last_details(const dl_context *, dl_detail_stats *);
/* Immutable trained model file; no teacher or HQ full-frame dependency.
 * A context owns scratch storage and must not be used concurrently.
 * Separate contexts may run concurrently. NULL returned on invalid model. */
dl_context *dl_create(const char *model_path);
/* Embed the generated model header; bytes are copied and validated at startup.
 */
dl_context *dl_create_from_memory(const void *bytes, size_t length);
void dl_destroy(dl_context *);
size_t dl_model_bytes(const dl_context *);
const char *dl_backend(void);
/* 0 auto, 1 serial, 2..8 requested row workers. Apple uses GCD; other builds
 * stay serial. A context must still have only one caller at a time. */
int dl_set_workers(dl_context *, unsigned workers);
/* Both formats keep each output 3x3 block within the per-channel RGB
 * minimum/maximum of its source 3x3 neighbourhood (edge-replicated borders).
 * BGRA8 bytes in/out; alpha is ignored on input and written as 255. This
 * format uploads directly to an ordinary BGRA8 Metal texture. */
int dl_scale3_bgra(dl_context *, const uint8_t *src, int width, int height,
                   size_t src_stride, uint8_t *dst, size_t dst_stride,
                   const dl_rect *regions, size_t count, dl_stats *stats);
/* Exact 3x RGB24; positive byte strides; non-overlapping input/output.
 * Regions are optional, half-open source-coordinate rectangles.
 * Status: 0 success, 1 invalid arguments/model, 2 allocation failure. */
int dl_scale3(dl_context *, const uint8_t *src, int width, int height,
              size_t src_stride, uint8_t *dst, size_t dst_stride,
              const dl_rect *regions, size_t count, dl_stats *stats);
#ifdef __cplusplus
}
#endif
#endif
