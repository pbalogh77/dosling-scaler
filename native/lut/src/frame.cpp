#include "model.hpp"
#if defined(DL_NEON) && (defined(__aarch64__) || defined(__ARM_NEON))
#include <arm_neon.h>
#define DL_SIMD 1
#endif
namespace lut {
#ifdef DL_SIMD
static uint32x4_t distance4(uint32x4_t a, uint32x4_t b) {
  auto mask = vdupq_n_u32(255);
  auto r =
      vsubq_s32(vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(a, 16), mask)),
                vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(b, 16), mask)));
  auto g = vsubq_s32(vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(a, 8), mask)),
                     vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(b, 8), mask)));
  auto bb = vsubq_s32(vreinterpretq_s32_u32(vandq_u32(a, mask)),
                      vreinterpretq_s32_u32(vandq_u32(b, mask)));
  return vreinterpretq_u32_s32(
      vmlaq_s32(vmlaq_s32(vmulq_s32(r, r), g, g), bb, bb));
}
#endif
void Frame::prepare(const uint8_t *src, int width, int height, size_t stride,
                    const dl_rect *regions, size_t nr, bool /* legacy_palette_neighbors */,
                    bool bgra) {
  w = width;
  h = height;
  pitch = w + 4;
  pixels.resize(size_t(pitch) * (h + 4));
  keys.resize(size_t(w) * h);
  yuv.resize(pixels.size());
  fallback_keys.resize(size_t(w) * h);
  threshold.resize(size_t(w) * h);
  text.assign(size_t(w) * h, 0);
  for (int y = 0; y < h; ++y) {
    auto row = src + size_t(y) * stride;
    int x = 0;
#ifdef DL_SIMD
    if (bgra) {
      for (; x + 4 <= w; x += 4)
        vst1q_u32(&pixels[size_t(y + 2) * pitch + x + 2],
                  vandq_u32(vld1q_u32(reinterpret_cast<const uint32_t *>(
                                row + x * 4)),
                            vdupq_n_u32(0xffffff)));
    } else {
      for (; x + 16 <= w; x += 16) {
        auto c = vld3q_u8(row + x * 3);
        uint8x16x4_t packed{{c.val[2], c.val[1], c.val[0], vdupq_n_u8(0)}};
        vst4q_u8(
            reinterpret_cast<uint8_t *>(&pixels[size_t(y + 2) * pitch + x + 2]),
            packed);
      }
    }
#endif
    for (; x < w; ++x) {
      const auto p = row + x * (bgra ? 4 : 3);
      pixels[size_t(y + 2) * pitch + x + 2] =
          (uint32_t(p[bgra ? 2 : 0]) << 16) | (uint32_t(p[1]) << 8) |
          p[bgra ? 0 : 2];
    }
    pixels[size_t(y + 2) * pitch] = pixels[size_t(y + 2) * pitch + 1] =
        pixels[size_t(y + 2) * pitch + 2];
    pixels[size_t(y + 2) * pitch + w + 2] =
        pixels[size_t(y + 2) * pitch + w + 3] =
            pixels[size_t(y + 2) * pitch + w + 1];
  }
  for (int i = 0; i < 2; ++i) {
    std::copy_n(&pixels[size_t(2) * pitch], pitch, &pixels[size_t(i) * pitch]);
    std::copy_n(&pixels[size_t(h + 1) * pitch], pitch,
                &pixels[size_t(h + 2 + i) * pitch]);
  }
  size_t yi = 0;
#ifdef DL_SIMD
  // Exact signed /1000 for this bounded RGB/YUV range, using widening integer
  // products. abs(n)*ceil(2^38/1000)>>38 equals abs(n)/1000 here.
  auto div1000 = [](int32x4_t n) {
    auto a = vreinterpretq_u32_s32(vabsq_s32(n));
    auto lo =
        vshrq_n_u64(vmull_u32(vget_low_u32(a), vdup_n_u32(274877907)), 38);
    auto hi =
        vshrq_n_u64(vmull_u32(vget_high_u32(a), vdup_n_u32(274877907)), 38);
    auto q = vreinterpretq_s32_u32(vcombine_u32(vmovn_u64(lo), vmovn_u64(hi)));
    return vbslq_s32(vcltq_s32(n, vdupq_n_s32(0)), vnegq_s32(q), q);
  };
  for (; yi + 4 <= pixels.size(); yi += 4) {
    auto c = vld1q_u32(&pixels[yi]), mask = vdupq_n_u32(255);
    auto r = vreinterpretq_s32_u32(vshrq_n_u32(c, 16)),
         g = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(c, 8), mask)),
         b = vreinterpretq_s32_u32(vandq_u32(c, mask));
    auto yy =
        div1000(vaddq_s32(vaddq_s32(vmulq_n_s32(r, 299), vmulq_n_s32(g, 587)),
                          vmulq_n_s32(b, 114)));
    auto u = vaddq_s32(
        div1000(vaddq_s32(vaddq_s32(vmulq_n_s32(r, -169), vmulq_n_s32(g, -331)),
                          vmulq_n_s32(b, 500))),
        vdupq_n_s32(128));
    auto v = vaddq_s32(
        div1000(vaddq_s32(vaddq_s32(vmulq_n_s32(r, 500), vmulq_n_s32(g, -419)),
                          vmulq_n_s32(b, -81))),
        vdupq_n_s32(128));
    vst1q_u32(&yuv[yi],
              vorrq_u32(vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(yy), 16),
                                  vshlq_n_u32(vreinterpretq_u32_s32(u), 8)),
                        vreinterpretq_u32_s32(v)));
  }
#endif
  for (size_t i = yi; i < pixels.size(); ++i) {
    auto c = rgb(pixels[i]);
    int yy = (299 * c.r + 587 * c.g + 114 * c.b) / 1000,
        u = (-169 * c.r - 331 * c.g + 500 * c.b) / 1000 + 128,
        v = (500 * c.r - 419 * c.g - 81 * c.b) / 1000 + 128;
    yuv[i] = (uint32_t(yy) << 16) | (uint32_t(u) << 8) | uint32_t(v);
  }
  for (size_t i = 0; i < nr; ++i) {
    auto r = regions[i];
    for (int y = r.y0; y < r.y1; ++y)
      std::fill(text.begin() + size_t(y) * w + r.x0,
                text.begin() + size_t(y) * w + r.x1, 1);
  }
  for (int y = 0; y < h; ++y) {
    int x = 0;
#ifdef DL_SIMD
    for (; x + 4 <= w; x += 4) {
      size_t p = size_t(y + 2) * pitch + x + 2;
      auto c = vld1q_u32(&pixels[p]);
      uint32x4_t ds[8], far = vdupq_n_u32(0);
      int j = 0;
      for (int i = 0; i < 9; ++i)
        if (i != 4) {
          ds[j] = distance4(c, vld1q_u32(&pixels[p + cy[i] * pitch + cx[i]]));
          far = vmaxq_u32(far, ds[j++]);
        }
      auto t = vmaxq_u32(vshrq_n_u32(far, 5), vdupq_n_u32(48)),
           key = vdupq_n_u32(0);
      for (int i = 0; i < 8; ++i)
        key = vorrq_u32(key,
                        vandq_u32(vcgtq_u32(ds[i], t), vdupq_n_u32(1u << i)));
      auto center_uv = vld1q_u32(&yuv[p]);
      auto hqkey = vdupq_n_u32(0);
      int bit = 0;
      for (int i = 0; i < 9; ++i)
        if (i != 4) {
          auto v = vld1q_u32(&yuv[p + cy[i] * pitch + cx[i]]);
          auto mask = vdupq_n_u32(255);
          auto delta = [&](uint32x4_t a, uint32x4_t b) {
            return vabdq_s32(vreinterpretq_s32_u32(a),
                             vreinterpretq_s32_u32(b));
          };
          auto yy = delta(vshrq_n_u32(center_uv, 16), vshrq_n_u32(v, 16)),
               u = delta(vandq_u32(vshrq_n_u32(center_uv, 8), mask),
                         vandq_u32(vshrq_n_u32(v, 8), mask)),
               vv = delta(vandq_u32(center_uv, mask), vandq_u32(v, mask));
          auto diff = vorrq_u32(vcgtq_s32(yy, vdupq_n_s32(48)),
                                vorrq_u32(vcgtq_s32(u, vdupq_n_s32(7)),
                                          vcgtq_s32(vv, vdupq_n_s32(6))));
          hqkey = vorrq_u32(hqkey, vandq_u32(diff, vdupq_n_u32(1u << bit++)));
        }
      uint32_t hqkeys[4];
      vst1q_u32(hqkeys, hqkey);
      for (int i = 0; i < 4; ++i)
        fallback_keys[size_t(y) * w + x + i] = uint8_t(hqkeys[i]);
      uint32_t kk[4];
      vst1q_u32(kk, key);
      vst1q_u32(&threshold[size_t(y) * w + x], t);
      for (int i = 0; i < 4; ++i)
        keys[size_t(y) * w + x + i] = uint8_t(kk[i]);
    }
#endif
    for (; x < w; ++x) {
      size_t p = size_t(y + 2) * pitch + x + 2;
      uint32_t ds[8], far = 0;
      int j = 0;
      for (int i = 0; i < 9; ++i)
        if (i != 4) {
          ds[j] = distance(pixels[p], pixels[p + cy[i] * pitch + cx[i]]);
          far = std::max(far, ds[j++]);
        }
      uint32_t t = std::max(48u, far >> 5), key = 0;
      for (int i = 0; i < 8; ++i)
        if (ds[i] > t)
          key |= 1u << i;
      keys[size_t(y) * w + x] = uint8_t(key);
      threshold[size_t(y) * w + x] = t;
      auto a = rgb(yuv[p]);
      uint8_t hqkey = 0;
      int bit = 0;
      for (int i = 0; i < 9; ++i)
        if (i != 4) {
          auto b = rgb(yuv[p + cy[i] * pitch + cx[i]]);
          if (std::abs(a.r - b.r) > 48 || std::abs(a.g - b.g) > 7 ||
              std::abs(a.b - b.b) > 6)
            hqkey |= uint8_t(1u << bit);
          ++bit;
        }
      fallback_keys[size_t(y) * w + x] = hqkey;
    }
  }
}
void Frame::core(int x, int y, uint32_t out[9]) const {
  size_t p = size_t(y + 2) * pitch + x + 2;
  for (int i = 0; i < 9; ++i)
    out[i] = pixels[p + cy[i] * pitch + cx[i]];
}
uint8_t Frame::outer(int x, int y, uint32_t t, int plan) const {
  size_t p = size_t(y + 2) * pitch + x + 2;
  uint8_t key = 0;
  if (plan) {
    constexpr int pairs[][2] = {{1, 3}, {1, 5}, {7, 3}, {7, 5}};
    for (int i = 0; i < 4; ++i)
      if (uint32_t(distance(
              pixels[p + cy[pairs[i][0]] * pitch + cx[pairs[i][0]]],
              pixels[p + cy[pairs[i][1]] * pitch + cx[pairs[i][1]]])) > t)
        key |= uint8_t(1 << i);
    for (int i = 0; i < 4; ++i)
      if (uint32_t(distance(pixels[p], pixels[p + ey[i] * pitch + ex[i]])) > t)
        key |= uint8_t(1 << (i + 4));
  } else
    for (int i = 0; i < 8; ++i)
      if (uint32_t(distance(pixels[p], pixels[p + ey[i] * pitch + ex[i]])) > t)
        key |= uint8_t(1 << i);
  return key;
}
uint16_t Frame::outer5(int x, int y, uint32_t t) const {
  const size_t p = size_t(y + 2) * pitch + x + 2;
#ifdef DL_SIMD
  const uint32x4_t samples[4] = {
      vld1q_u32(&pixels[p - 2 * pitch - 2]),
      {pixels[p - 2 * pitch + 2], pixels[p - pitch - 2], pixels[p - pitch + 2],
       pixels[p - 2]},
      {pixels[p + 2], pixels[p + pitch - 2], pixels[p + pitch + 2],
       pixels[p + 2 * pitch - 2]},
      vld1q_u32(&pixels[p + 2 * pitch - 1])};
  uint16_t key = 0;
  const uint32x4_t weights = {1, 2, 4, 8};
  for (int i = 0; i < 4; ++i) {
    auto bits =
        vandq_u32(vcgtq_u32(distance4(vdupq_n_u32(pixels[p]), samples[i]),
                            vdupq_n_u32(t)),
                  weights);
    uint32_t value = vgetq_lane_u32(bits, 0) | vgetq_lane_u32(bits, 1) |
                     vgetq_lane_u32(bits, 2) | vgetq_lane_u32(bits, 3);
    key |= uint16_t(value << (i * 4));
  }
  return key;
#else
  uint16_t key = 0;
  int bit = 0;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx)
      if (std::abs(dy) == 2 || std::abs(dx) == 2) {
        if (uint32_t(distance(pixels[p], pixels[p + dy * pitch + dx])) > t)
          key |= uint16_t(1u << bit);
        ++bit;
      }
  return key;
#endif
}
Frame::BankSelection Frame::bank(int x, int y, const uint32_t *c,
                                 bool plane_exception) const {
  if (text[size_t(y) * w + x])
    return {1};
  if (!plane_exception)
    return {0};
  // Bank2 now serves only ordinary binary plane/diagonal exceptions.
  // Checker interiors and checker/material boundaries use the graphics bank.
  switch (keys[size_t(y) * w + x]) {
  // Same strict two-endpoint rule matching, for ordinary plane AA.
  // Binary45degree exceptions also require actual two-color surfaces.
  case 11: case 22: case 104: case 208:
  case 3: case 6: case 7: case 9: case 15: case 20: case 23: case 40: case 41: case 43: case 96: case 105: case 144: case 148: case 150: case 192: case 212: case 224: case 232: case 240:
    break;
  default:
    return {0};
  }
  // Require25actual source samples, not clamped padding.
  if (x<2 || y<2 || x>=w-2 || y>=h-2)
    return {0};
  // A cared nonzero contrast bit identifies the other endpoint.
  const int endpoint_bit=__builtin_ctz(unsigned(keys[size_t(y)*w+x]));
  uint32_t other=c[endpoint_bit+(endpoint_bit>=4)];
  if (other == c[4])
    return {0};
#if defined(DL_SIMD) && defined(__aarch64__)
  {
    const auto center4=vdupq_n_u32(c[4]), other4=vdupq_n_u32(other);
    for (int i : {0,4}) {
      const auto values=vld1q_u32(c+i);
      const auto valid=vorrq_u32(vceqq_u32(values,center4),vceqq_u32(values,other4));
      if (vminvq_u32(valid)!=UINT32_MAX) return {0};
    }
    if (c[8]!=c[4] && c[8]!=other) return {0};
  }
#else
  {
    for (int i=0;i<9;++i)
      if (c[i]!=c[4] && c[i]!=other) return {0};
  }
#endif
  // Binary contrast bits alone can confuse a third shade with either color.
  // Check exact endpoint colors only on these rare eligible edge prefixes.
  size_t center = size_t(y + 2) * pitch + x + 2;
  uint16_t key = 0;
#if defined(DL_SIMD) && defined(__aarch64__)
  const uint32x4_t samples[4] = {
      vld1q_u32(&pixels[center - 2 * pitch - 2]),
      {pixels[center - 2 * pitch + 2], pixels[center - pitch - 2],
       pixels[center - pitch + 2], pixels[center - 2]},
      {pixels[center + 2], pixels[center + pitch - 2],
       pixels[center + pitch + 2], pixels[center + 2 * pitch - 2]},
      vld1q_u32(&pixels[center + 2 * pitch - 1])};
  const auto a4 = vdupq_n_u32(c[4]), b4 = vdupq_n_u32(other);
  const uint32x4_t weights = {1, 2, 4, 8};
  for (int i = 0; i < 4; ++i) {
    auto a = vceqq_u32(samples[i], a4), b = vceqq_u32(samples[i], b4);
    if (vminvq_u32(vorrq_u32(a, b)) != UINT32_MAX)
      return {0};
    key |= uint16_t(vaddvq_u32(vandq_u32(b, weights)) << (i * 4));
  }
#else
  int bit = 0;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx)
      if (std::abs(dx) == 2 || std::abs(dy) == 2) {
        uint32_t pixel = pixels[center + dy * pitch + dx];
        if (pixel != c[4] && pixel != other)
          return {0};
        if (pixel == other)
          key |= uint16_t(1u << bit);
        ++bit;
      }
#endif
  // This path has an exact two-color core and a nonzero inner key. Its color
  // distance therefore exceeds the same adaptive threshold used by outer5.
  // Reuse endpoint equality as the contrast bit; no second RGB-distance pass.
  return {2, true, key};
}
} // namespace lut
