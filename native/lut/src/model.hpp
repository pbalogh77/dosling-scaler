#pragma once
#include "dos_lut.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
namespace lut {
constexpr uint32_t VERSION = 4, BANKS = 3;
constexpr uint16_t NO_EXT = 65535;
constexpr uint8_t HQ = 255;
struct Op {
  uint8_t a, b, c, wa, wb, wc, kind, reserved;
};
struct Block {
  std::array<uint8_t, 9> op;
  bool operator<(const Block &b) const { return op < b.op; }
};
struct Entry {
  uint16_t recipe, extended;
};
struct Leaf5 {
  uint16_t key, recipe;
};
struct Rule5 {
  uint16_t value, mask, recipe;
};
struct Exception5 {
  std::vector<Leaf5> leaves;
  std::vector<Rule5> masked;
  // Sparse tables need only 16 bins; dense ones retain 256. Derived at load,
  // never serialized. uint32_t offsets also cover all 65,536 exact keys.
  std::vector<uint32_t> buckets;
  unsigned bucket_shift = 12;
  void index();
  void compress_masks();
  uint16_t lookup(uint16_t) const;
};
struct Model {
  uint32_t version = 3; // Keep the historical trainer's format stable.
  std::vector<Op> ops;
  std::vector<Block> blocks;
  std::vector<uint8_t>
      flags; // Derived: bit 0 all-center, bit 1 all-HQ, bit 2 uniform color op; bit 3 rounded contour with direct source-color borrowing.
  std::array<Entry, BANKS * 256> primary;
  std::vector<std::array<uint16_t, 256>> extended;
  std::vector<Exception5> exceptions;
};
static constexpr int cy[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1},
                     cx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
static constexpr int ey[8] = {-2, 0, 2, 0, -2, -2, 2, 2},
                     ex[8] = {0, 2, 0, -2, -2, 2, -2, 2};
struct Color {
  int r, g, b;
};
inline Color rgb(uint32_t c) {
  return {int((c >> 16) & 255), int((c >> 8) & 255), int(c & 255)};
}
inline uint32_t packed(Color c) {
  return (uint32_t(c.r) << 16) | (uint32_t(c.g) << 8) | uint32_t(c.b);
}
inline int distance(uint32_t a, uint32_t b) {
  Color x = rgb(a), y = rgb(b);
  int r = x.r - y.r, g = x.g - y.g, bb = x.b - y.b;
  return r * r + g * g + bb * bb;
}
inline int floor_div(int a, int b) {
  int q = a / b, r = a % b;
  return q - (r < 0);
}
inline int clip(int a, int lo = 0, int hi = 255) {
  return std::max(lo, std::min(a, hi));
}
// Existing position-independent copy/blend arithmetic. AA and HQ instructions
// are positional and must never use this ordinary-color shortcut.
inline uint32_t color_op(const Op &op, const uint32_t *c) {
  if (op.wa == 64)
    return c[op.a];
  uint32_t bias = op.reserved ? 0u : 32u, a = c[op.a], b = c[op.b], d = c[op.c];
  // R/B occupy separate 16-bit lanes; weights sum to 64, preventing carries.
  constexpr uint32_t RB = 0x00ff00ff, G = 0x0000ff00;
  uint32_t rb = ((a & RB)*op.wa + (b & RB)*op.wb + (d & RB)*op.wc +
                 (bias << 16) + bias) >> 6;
  uint32_t gg = ((a & G)*op.wa + (b & G)*op.wb + (d & G)*op.wc +
                 (bias << 8)) >> 6;
  return (rb & RB) | (gg & G);
}
std::vector<Op> operations();
Model load(const char *);
Model load_memory(const void *, size_t);
void save(const Model &, const char *);
size_t model_bytes(const Model &);
// Sparse prepared AA metadata. No model-format or public C ABI change.
struct AaPlane {
  size_t index = 0;
  uint32_t fg = 0, bg = 0, reciprocal = 0;
  int16_t s = 0, h2 = 0;
  uint16_t error = 0, total = 0;
  uint8_t axis = 0;
  bool eligible = true;
};
struct AaTie { size_t index; std::array<uint32_t,9> pixels; };
struct Frame {
  struct BankSelection {
    uint8_t index = 0;
    bool outer_ready = false;
    uint16_t outer = 0;
  };
  int w = 0, h = 0, pitch = 0;
  std::vector<uint32_t> pixels, threshold, yuv;
  std::vector<uint8_t> keys, text, fallback_keys, ufo_rules;
  std::vector<AaPlane> aa_planes;
  std::vector<AaTie> aa_ties;
  void prepare(const uint8_t *, int, int, size_t, const dl_rect *, size_t,
               bool legacy_palette_neighbors = false, bool bgra = false);
  void core(int x, int y, uint32_t out[9]) const;
  uint8_t outer(int x, int y, uint32_t threshold, int plan = 0) const;
  uint16_t outer5(int x, int y, uint32_t threshold) const;
  BankSelection bank(int x, int y, const uint32_t core[9],
                     bool plane_exception = false) const;
};
void warm_stencils();
void prepare_ufo(Frame &);
void collect_aa_plane(Frame &,int,int,bool hinted_only = true);
void finish_aa_planes(Frame &,const std::vector<size_t> &square_junctions);
int aa_plane_block(const Frame &,int,int,uint8_t,uint32_t,uint32_t *);
int ufo_block(const Frame &, int x, int y, const uint32_t *core, uint32_t *out);
bool aa_block(int, const uint32_t *, uint32_t *, bool text_stencil = false);
void evaluate(const Model &, const Block &, const uint32_t core[9],
              uint32_t output[9], bool *used_hq = nullptr,
              const uint32_t *yuv = nullptr, uint8_t fallback_key = 0);
struct Context {
  Model model;
  Frame frame;
  size_t bytes = 0;
  dl_detail_stats details{};
  unsigned workers = 0;
};
} // namespace lut
struct dl_context {
  lut::Context impl;
};
extern "C" void ds_hq3x_block(const uint32_t *, uint32_t *);

extern "C" void ds_hq3x_block_cached(const uint32_t *, const uint32_t *, int,
                                     uint32_t *);
