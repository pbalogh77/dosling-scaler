#include "model.hpp"
#include <sstream>
namespace lut {
static int specificity(uint16_t mask) {
  return __builtin_popcount(unsigned(mask));
}
static bool rule_order(const Rule5 &a, const Rule5 &b) {
  int sa = specificity(a.mask), sb = specificity(b.mask);
  if (sa != sb)
    return sa > sb;
  if (a.mask != b.mask)
    return a.mask > b.mask;
  if (a.value != b.value)
    return a.value < b.value;
  return a.recipe < b.recipe;
}
void Exception5::index() {
  std::sort(leaves.begin(), leaves.end(),
            [](auto a, auto b) { return a.key < b.key; });
  std::sort(masked.begin(), masked.end(), rule_order);
  bucket_shift = leaves.size() <= 256 ? 12 : 8;
  const unsigned bins = 1u << (16 - bucket_shift);
  buckets.resize(bins + 1);
  size_t i = 0;
  for (unsigned high = 0; high <= bins; ++high) {
    while (i < leaves.size() && (unsigned(leaves[i].key) >> bucket_shift) < high)
      ++i;
    buckets[high] = uint32_t(i);
  }
}
void Exception5::compress_masks() {
  std::vector<Rule5> rules;
  for (auto l : leaves)
    rules.push_back({l.key, 65535, l.recipe});
  // Merge two complete, disjoint cubes only when both recipes are identical.
  // Their union is exact: no unseen pattern or conflicting fitted recipe is
  // silently absorbed by a don't-care bit.
  bool changed = true;
  while (changed) {
    changed = false;
    std::sort(rules.begin(), rules.end(), rule_order);
    std::map<std::tuple<uint16_t, uint16_t, uint16_t>, size_t> index;
    for (size_t i = 0; i < rules.size(); ++i)
      index[{rules[i].recipe, rules[i].mask, rules[i].value}] = i;
    std::vector<bool> used(rules.size());
    std::vector<Rule5> next;
    for (size_t i = 0; i < rules.size(); ++i) {
      if (used[i])
        continue;
      auto a = rules[i];
      bool merged = false;
      for (int bit = 0; bit < 16; ++bit)
        if (a.mask & (1u << bit)) {
          auto it =
              index.find({a.recipe, a.mask, uint16_t(a.value ^ (1u << bit))});
          if (it != index.end() && it->second > i && !used[it->second]) {
            uint16_t mask = uint16_t(a.mask & ~(1u << bit));
            next.push_back({uint16_t(a.value & mask), mask, a.recipe});
            used[i] = used[it->second] = true;
            changed = merged = true;
            break;
          }
        }
      if (!merged)
        next.push_back(a);
    }
    rules = std::move(next);
  }
  // Bound the masked slow path. Expand extra cubes back to exact entries.
  leaves.clear();
  masked.clear();
  std::sort(rules.begin(), rules.end(), [](auto a, auto b) {
    int sa = specificity(a.mask), sb = specificity(b.mask);
    return sa != sb ? sa < sb : rule_order(a, b);
  });
  for (auto r : rules) {
    if (r.mask != 65535 && masked.size() < 64) {
      masked.push_back(r);
      continue;
    }
    uint16_t free = uint16_t(~r.mask), subset = free;
    while (true) {
      leaves.push_back({uint16_t(r.value | subset), r.recipe});
      if (!subset)
        break;
      subset = uint16_t((subset - 1) & free);
    }
  }
  index();
}
uint16_t Exception5::lookup(uint16_t key) const {
  const unsigned bin = key >> bucket_shift;
  size_t lo = buckets[bin], hi = buckets[bin + 1], end = hi;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (leaves[mid].key < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo < end && leaves[lo].key == key)
    return leaves[lo].recipe;
  for (auto rule : masked)
    if ((key & rule.mask) == rule.value)
      return rule.recipe;
  return NO_EXT;
}
size_t model_bytes(const Model &m) {
  size_t n = 28 + m.ops.size() * 8 + m.blocks.size() * 9 + BANKS * 256 * 4;
  if (m.version < 4)
    return n + m.extended.size() * 512;
  for (const auto &e : m.exceptions)
    n += 4 + (e.leaves.size() + e.masked.size()) * 6;
  return n;
}

std::vector<Op> operations() {
  std::vector<Op> ops;
  for (int a = 0; a < 9; ++a)
    ops.push_back({uint8_t(a), 4, 4, 64, 0, 0, 0, 0});
  for (int a = 0; a < 9; ++a)
    if (a != 4)
      for (int t = 4; t < 64; t += 4)
        ops.push_back(
            {4, uint8_t(a), 4, uint8_t(64 - t), uint8_t(t), 0, 0, 0}); // 129
  // Adjacent ring pairs can smooth diagonals without importing a third hue.
  constexpr int ring[] = {0, 1, 2, 5, 8, 7, 6, 3};
  for (int i = 0; i < 8; ++i)
    for (int q : {16, 32, 48})
      ops.push_back({uint8_t(ring[i]), uint8_t(ring[(i + 1) % 8]), 4,
                     uint8_t(q), uint8_t(64 - q), 0, 0, 0});
  constexpr int pairs[][2] = {{1, 3}, {1, 5}, {7, 3}, {7, 5},
                              {0, 8}, {2, 6}, {1, 7}, {3, 5}};
  for (auto &p : pairs) {
    ops.push_back({uint8_t(p[0]), uint8_t(p[1]), 4, 32, 32, 0, 0, 0});
    for (int center : {16, 32, 48})
      ops.push_back({4, uint8_t(p[0]), uint8_t(p[1]), uint8_t(center),
                     uint8_t((64 - center) / 2), uint8_t((64 - center) / 2), 0,
                     0});
  }
  for (int k = 0; k < 4; ++k)
    ops.push_back({0, 0, 0, 0, 0, 0, uint8_t(k + 1), 0});
  // Exact truncating HQx weights are available as learned operators too.
  for (int a = 0; a < 9; ++a)
    if (a != 4)
      for (int t : {8, 16, 32, 48, 56})
        ops.push_back({4, uint8_t(a), 4, uint8_t(64 - t), uint8_t(t), 0, 0, 1});
  for (int k = 0; k < 4; ++k) {
    auto &p = pairs[k];
    for (int center : {8, 32})
      ops.push_back({4, uint8_t(p[0]), uint8_t(p[1]), uint8_t(center),
                     uint8_t((64 - center) / 2), uint8_t((64 - center) / 2), 0,
                     1});
    ops.push_back({uint8_t(p[0]), uint8_t(p[1]), 4, 32, 32, 0, 0, 1});
  }
  if (ops.size() >= 255)
    throw std::runtime_error("Operator dictionary overflow");
  return ops;
}
static uint32_t u32(std::istream &f) {
  uint8_t b[4];
  if (!f.read((char *)b, 4))
    throw std::runtime_error("Truncated model");
  return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) |
         (uint32_t(b[3]) << 24);
}
static void u32(std::ostream &f, uint32_t n) {
  uint8_t b[4] = {uint8_t(n), uint8_t(n >> 8), uint8_t(n >> 16),
                  uint8_t(n >> 24)};
  f.write((char *)b, 4);
}
static uint16_t u16(std::istream &f) {
  int a = f.get(), b = f.get();
  if (a < 0 || b < 0)
    throw std::runtime_error("Truncated model");
  return uint16_t(a | (b << 8));
}
static void u16(std::ostream &f, uint16_t n) {
  f.put(char(n & 255));
  f.put(char(n >> 8));
}
static Model read(std::istream &f) {
  char magic[8];
  if (!f.read(magic, 8) || std::memcmp(magic, "DOSLUT3\0", 8))
    throw std::runtime_error("Invalid model");
  uint32_t version = u32(f);
  if (version < 1 || version > VERSION || u32(f) != BANKS)
    throw std::runtime_error("Invalid model");
  uint32_t no = u32(f), nb = u32(f), ne = u32(f);
  if (no == 0 || no >= 255 || nb == 0 || nb >= 65535 || ne > BANKS * 256)
    throw std::runtime_error("Invalid model counts");
  Model m;
  m.version = version;
  m.ops.resize(no);
  m.blocks.resize(nb);
  if (version < 4)
    m.extended.resize(ne);
  else
    m.exceptions.resize(ne);
  if (!f.read((char *)m.ops.data(), no * sizeof(Op)))
    throw std::runtime_error("Truncated model");
  for (auto &o : m.ops)
    if (o.reserved > 1 || o.kind > 4 ||
        (o.kind == 0 &&
         (o.a > 8 || o.b > 8 || o.c > 8 || o.wa + o.wb + o.wc != 64)))
      throw std::runtime_error("Invalid operator");
  for (auto &b : m.blocks) {
    if (!f.read((char *)b.op.data(), 9))
      throw std::runtime_error("Truncated model");
    for (auto o : b.op)
      if (o != HQ && o >= no)
        throw std::runtime_error("Invalid recipe");
  }
  for (auto &e : m.primary) {
    e.recipe = u16(f);
    e.extended = u16(f);
    if (e.recipe >= nb ||
        (e.extended != NO_EXT &&
         (version < 4 ? (e.extended & 0x7fff) : e.extended) >= ne))
      throw std::runtime_error("Invalid index");
  }
  for (auto &table : m.extended)
    for (auto &r : table) {
      r = u16(f);
      if (r != NO_EXT && r >= nb)
        throw std::runtime_error("Invalid extension");
    }
  uint32_t total = 0;
  for (auto &e : m.exceptions) {
    uint32_t n = u32(f);
    if (!n || n > 65536 || n > 65536 - total)
      throw std::runtime_error("Invalid exception count");
    total += n;
    std::set<uint16_t> exact_keys;
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t value = u16(f), mask = u16(f), recipe = u16(f);
      if ((value & uint16_t(~mask)) || recipe >= nb)
        throw std::runtime_error("Invalid masked exception");
      if (mask == 65535) {
        if (!exact_keys.insert(value).second)
          throw std::runtime_error("Duplicate exact key");
        e.leaves.push_back({value, recipe});
      } else {
        if (e.masked.size() >= 64)
          throw std::runtime_error("Too many masked rules");
        for (auto r : e.masked)
          if (specificity(r.mask) == specificity(mask) && r.recipe != recipe &&
              !((r.value ^ value) & r.mask & mask))
            throw std::runtime_error("Ambiguous masked rules");
        e.masked.push_back({value, mask, recipe});
      }
    }
    e.index();
  }
  if (f.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Trailing model data");
  m.flags.resize(m.blocks.size());
  for (size_t i = 0; i < m.blocks.size(); ++i) {
    bool nn = true, hq = true, imports = false, rounded = true;
    int first = m.blocks[i].op[0];
    bool uniform = first != HQ && !m.ops[first].kind;
    for (int p = 0; p < 9; ++p) {
      int id = m.blocks[i].op[p];
      bool center = false;
      if (id != HQ) {
        auto o = m.ops[id];
        center = !o.kind && (!o.wa || o.a == 4) && (!o.wb || o.b == 4) &&
                 (!o.wc || o.c == 4);
      }
      imports |= id != HQ && !m.ops[id].kind && !center;
      rounded &= id != HQ && !m.ops[id].kind && !m.ops[id].reserved;
      nn &= center;
      hq &= id == HQ || (p == 4 && center);
      uniform &= id == first;
    }
    m.flags[i] = uint8_t((nn ? 1 : hq ? 2 : uniform ? 4 : 0) | (imports && rounded ? 8 : 0));
  }
  return m;
}
Model load(const char *path) {
  std::ifstream f(path, std::ios::binary);
  return read(f);
}
Model load_memory(const void *bytes, size_t n) {
  if (!bytes || n > 1024 * 1024)
    throw std::runtime_error("Invalid model buffer");
  std::string data(static_cast<const char *>(bytes), n);
  std::istringstream f(data, std::ios::binary);
  return read(f);
}
void save(const Model &m, const char *path) {
  std::ofstream f(path, std::ios::binary);
  f.write("DOSLUT3\0", 8);
  u32(f, m.version);
  u32(f, BANKS);
  u32(f, uint32_t(m.ops.size()));
  u32(f, uint32_t(m.blocks.size()));
  u32(f, uint32_t(m.version < 4 ? m.extended.size() : m.exceptions.size()));
  f.write((const char *)m.ops.data(), m.ops.size() * sizeof(Op));
  for (auto &b : m.blocks)
    f.write((const char *)b.op.data(), 9);
  for (auto &e : m.primary) {
    u16(f, e.recipe);
    u16(f, e.extended);
  }
  for (auto &table : m.extended)
    for (auto r : table)
      u16(f, r);
  for (const auto &e : m.exceptions) {
    u32(f, uint32_t(e.leaves.size() + e.masked.size()));
    for (auto leaf : e.leaves) {
      u16(f, leaf.key);
      u16(f, 65535);
      u16(f, leaf.recipe);
    }
    for (auto rule : e.masked) {
      u16(f, rule.value);
      u16(f, rule.mask);
      u16(f, rule.recipe);
    }
  }
  if (!f)
    throw std::runtime_error("Cannot write model");
}
static std::array<int, 9> aa_coverage(int a) {
  if (a < 128) {
    auto c = aa_coverage(256 - a);
    std::reverse(c.begin(), c.end());
    for (auto &v : c)
      v = 256 - v;
    return c;
  }
  auto areas = [](int t) {
    std::array<int, 9> c;
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        int q = clip(t - (x + y) * 256, 0, 512);
        c[y * 3 + x] = q <= 256 ? 256 - (q * q + 256) / 512
                                : ((512 - q) * (512 - q) + 256) / 512;
      }
    return c;
  };
  int lo = 0, hi = 1537;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    auto c = areas(mid);
    int sum = 0;
    for (int v : c)
      sum += v;
    if (sum > a * 9)
      lo = mid + 1;
    else
      hi = mid;
  }
  auto c = areas(lo), d = areas(std::max(0, lo - 1));
  int ec = 0, ed = 0;
  for (int v : c)
    ec += v;
  for (int v : d)
    ed += v;
  return std::abs(ed - a * 9) <= std::abs(ec - a * 9) ? d : c;
}
static const auto &aa_table() {
  static const auto table = []() {
    std::array<std::array<int, 9>, 257> t;
    for (int a = 0; a <= 256; ++a)
      t[a] = aa_coverage(a);
    return t;
  }();
  return table;
}
bool aa_block(int kind, const uint32_t *c, uint32_t *out, bool text_stencil) {
  int dy = (kind <= 2) ? 1 : -1, dx = (kind == 1 || kind == 3) ? 1 : -1;
  auto index = [](int y, int x) { return (y + 1) * 3 + x + 1; };
  uint32_t h = c[index(0, dx)], v = c[index(dy, 0)], oh = c[index(0, -dx)],
           ov = c[index(-dy, 0)];
  if (distance(h, v) > 144 || distance(oh, ov) > 144)
    return false;
  Color hh = rgb(h), vv = rgb(v), o = rgb(oh), p = rgb(ov),
        fg{(hh.r + vv.r + 1) / 2, (hh.g + vv.g + 1) / 2, (hh.b + vv.b + 1) / 2},
        bg{(o.r + p.r + 1) / 2, (o.g + p.g + 1) / 2, (o.b + p.b + 1) / 2},
        center = rgb(c[4]);
  int dr = fg.r - bg.r, dg = fg.g - bg.g, db = fg.b - bg.b,
      n = dr * dr + dg * dg + db * db;
  if (n < 1800)
    return false;
  int a = floor_div(((center.r - bg.r) * dr + (center.g - bg.g) * dg +
                     (center.b - bg.b) * db) *
                            256 +
                        n / 2,
                    n);
  // Only the hardcoded text pass supplies stronger 5x5 endpoint checks,
  // contrast gates and continuation of both surfaces for extreme ratios.
  // Learned AA operators retain their historical admission in all banks.
  if (a < (text_stencil ? 16 : 48) || a > (text_stencil ? 248 : 220))
    return false;
  Color recon{bg.r + floor_div(a * dr + 128, 256),
              bg.g + floor_div(a * dg + 128, 256),
              bg.b + floor_div(a * db + 128, 256)};
  // Only source-validated text stencils get extended
  // mixture tolerance. Learned operators retain their strict historic gate.
  const int residual_limit = text_stencil ? std::min(1024,std::max(144,n/64)) : 144;
  if (distance(packed(recon), c[4]) > residual_limit)
    return false;
  auto &cov = aa_table()[a];
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) {
      int q = cov[(dy < 0 ? 2 - y : y) * 3 + (dx < 0 ? 2 - x : x)];
      out[y * 3 + x] = packed({clip(bg.r + floor_div(q * dr + 128, 256)),
                               clip(bg.g + floor_div(q * dg + 128, 256)),
                               clip(bg.b + floor_div(q * db + 128, 256))});
    }
  return true;
}
void evaluate(const Model &m, const Block &block, const uint32_t *c,
              uint32_t *out, bool *used_hq, const uint32_t *yuv,
              uint8_t fallback_key) {
  uint32_t hq[9], corners[4][9];
  bool ready = false, done[4] = {};
  if (used_hq)
    *used_hq = false;
  auto fallback = [&] {
    if (!ready) {
      if (yuv)
        ds_hq3x_block_cached(c, yuv, fallback_key, hq);
      else
        ds_hq3x_block(c, hq);
      ready = true;
      if (used_hq)
        *used_hq = true;
    }
  };
  for (int pos = 0; pos < 9; ++pos) {
    int id = block.op[pos];
    if (id == HQ) {
      fallback();
      out[pos] = hq[pos];
      continue;
    }
    auto op = m.ops[id];
    if (op.kind) {
      int k = op.kind - 1;
      if (!done[k]) {
        if (!aa_block(op.kind, c, corners[k])) {
          fallback();
          std::memcpy(corners[k], hq, sizeof hq);
        }
        done[k] = true;
      }
      out[pos] = corners[k][pos];
      continue;
    }
    out[pos] = color_op(op, c);
  }
}
} // namespace lut
