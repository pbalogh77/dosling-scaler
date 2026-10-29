#pragma once
#include "model.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
using namespace lut;
using I = int64_t;
constexpr int K = 8;
constexpr int NGRAM = 36, NCROSS = 72, NY = 9, NHQ = 9, NAA = 36,
              DIM = NGRAM + NCROSS + NY + NHQ + NAA;
struct Stats {
  std::array<uint8_t, 9 * K> candidates{};
  std::array<uint64_t, 9 * K> exact{};
  uint64_t validation_base = 0, validation_extended = 0;
  uint64_t n = 0;
  std::array<I, DIM> sum{};
  void add(const std::array<I, DIM> &v) {
    ++n;
    for (int i = 0; i < DIM; ++i)
      sum[i] += v[i];
  }
};
struct Dataset {
  std::array<Stats, BANKS * 256> base;
  std::unordered_map<uint32_t, Stats> extended;
  uint64_t samples = 0, frames = 0;
};
int gram_index(int i, int j) {
  if (i > j)
    std::swap(i, j);
  return i * 8 - i * (i - 1) / 2 + j - i;
}
int feature_index(int i) { return i < 4 ? i : i - 1; }
I dotc(Color a, Color b) { return I(a.r) * b.r + I(a.g) * b.g + I(a.b) * b.b; }
Color sub(Color a, Color b) { return {a.r - b.r, a.g - b.g, a.b - b.b}; }
std::array<I, DIM> contribution(const uint32_t *c, const uint32_t *y) {
  std::array<I, DIM> f{};
  Color center = rgb(c[4]);
  Color x[8], target[9];
  for (int i = 0; i < 9; ++i) {
    if (i != 4)
      x[feature_index(i)] = sub(rgb(c[i]), center);
    target[i] = sub(rgb(y[i]), center);
  }
  int k = 0;
  for (int i = 0; i < 8; ++i)
    for (int j = i; j < 8; ++j)
      f[k++] = dotc(x[i], x[j]);
  for (int p = 0; p < 9; ++p)
    for (int i = 0; i < 8; ++i)
      f[k++] = dotc(target[p], x[i]);
  for (auto t : target)
    f[k++] = dotc(t, t);
  uint32_t hq[9];
  ds_hq3x_block(c, hq);
  for (int p = 0; p < 9; ++p)
    f[k++] = distance(hq[p], y[p]);
  for (int a = 1; a <= 4; ++a) {
    uint32_t pred[9];
    if (!aa_block(a, c, pred))
      std::copy_n(hq, 9, pred);
    for (int p = 0; p < 9; ++p)
      f[k++] = distance(pred[p], y[p]);
  }
  return f;
}
static uint32_t hash32(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352d;
  v ^= v >> 15;
  v *= 0x846ca68b;
  return v ^ (v >> 16);
}
static void transform(uint32_t *c, uint32_t *y, int rot) {
  uint32_t out[17], target[9];
  for (int i = 0; i < 17; ++i) {
    int yy = i < 9 ? cy[i] : ey[i - 9], xx = i < 9 ? cx[i] : ex[i - 9];
    if (rot & 4)
      xx = -xx;
    for (int r = 0; r < (rot & 3); ++r) {
      int t = yy;
      yy = xx;
      xx = -t;
    }
    int j;
    if (i < 9)
      j = (yy + 1) * 3 + xx + 1;
    else {
      j = 9;
      while (ey[j - 9] != yy || ex[j - 9] != xx)
        ++j;
    }
    out[j] = c[i];
    if (i < 9)
      target[j] = y[i];
  }
  std::copy_n(out, 17, c);
  std::copy_n(target, 9, y);
}
template <class Function>
static void iterate(const std::string &list, bool augment, Function fn) {
  std::ifstream lines(list);
  if (!lines)
    throw std::runtime_error("Cannot read dataset list");
  uint64_t frames = 0;
  std::string path;
  while (std::getline(lines, path)) {
    if (path.empty())
      continue;
    std::ifstream f(path, std::ios::binary);
    uint32_t w, h, nr, seed;
    f.read((char *)&w, 4);
    f.read((char *)&h, 4);
    f.read((char *)&nr, 4);
    f.read((char *)&seed, 4);
    if (!f || w > 4096 || h > 4096 || !w || !h || nr > 65536)
      throw std::runtime_error("Invalid dataset header");
    std::vector<uint8_t> src(size_t(w) * h * 3), target(size_t(w) * h * 27);
    std::vector<dl_rect> regions(nr);
    f.read((char *)src.data(), src.size());
    f.read((char *)target.data(), target.size());
    f.read((char *)regions.data(), regions.size() * sizeof(dl_rect));
    if (!f)
      throw std::runtime_error("Truncated dataset");
    Frame frame;
    frame.prepare(src.data(), int(w), int(h), w * 3, regions.data(),
                  regions.size(), true);
    for (int y = 0; y < int(h); ++y)
      for (int x = 0; x < int(w); ++x) {
        uint32_t hash = hash32(uint32_t(y * w + x) ^ seed);
        size_t pixel = size_t(y) * w + x;
        uint32_t quota = frame.text[pixel] ? 16000 : 5000;
        if (hash % (w * h) >= quota)
          continue;
        uint32_t c[17], t[9];
        frame.core(x, y, c);
        int bank = frame.bank(x, y, c).index;
        size_t source = size_t(y + 2) * frame.pitch + x + 2;
        for (int i = 0; i < 8; ++i)
          c[i + 9] = frame.pixels[source + ey[i] * frame.pitch + ex[i]];
        for (int oy = 0; oy < 3; ++oy)
          for (int ox = 0; ox < 3; ++ox) {
            auto p = &target[(size_t(y * 3 + oy) * w * 3 + x * 3 + ox) * 3];
            t[oy * 3 + ox] =
                (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | p[2];
          }
        if (augment)
          transform(c, t, int(hash >> 29));
        uint32_t maxd = 0;
        for (int i = 0; i < 9; ++i)
          maxd = std::max(maxd, uint32_t(distance(c[i], c[4])));
        uint32_t threshold = std::max(48u, maxd >> 5), key = 0, bit = 1;
        for (int i = 0; i < 9; ++i)
          if (i != 4) {
            if (uint32_t(distance(c[i], c[4])) > threshold)
              key |= bit;
            bit <<= 1;
          }
        uint32_t outer = 0;
        for (int i = 0; i < 8; ++i)
          if (uint32_t(distance(c[i + 9], c[4])) > threshold)
            outer |= 1u << i;
        uint32_t prefix = bank * 256 + key;
        uint32_t relations = 0;
        constexpr int pairs[][2] = {{1, 3}, {1, 5}, {7, 3}, {7, 5}};
        for (int i = 0; i < 4; ++i)
          if (uint32_t(distance(c[pairs[i][0]], c[pairs[i][1]])) > threshold)
            relations |= 1u << i;
        for (int i = 0; i < 4; ++i)
          if (uint32_t(distance(c[i + 9], c[4])) > threshold)
            relations |= 1u << (i + 4);
        fn(prefix, outer, relations, c, t);
      }
    if (++frames % 50 == 0)
      std::cerr << frames << " frames scanned\n";
  }
}
static void accumulate(Dataset &data, const std::string &list, bool augment) {
  iterate(list, augment,
          [&](uint32_t prefix, uint32_t outer, uint32_t relations,
              const uint32_t *c, const uint32_t *t) {
            auto value = contribution(c, t);
            data.base[prefix].add(value);
            data.extended[prefix * 256 + outer].add(value);
            data.extended[(BANKS * 256 + prefix) * 256 + relations].add(value);
            ++data.samples;
          });
  std::ifstream f(list);
  std::string line;
  while (std::getline(f, line))
    if (!line.empty())
      ++data.frames;
}

I cost(const Stats &s, int p, int id, const std::vector<Op> &ops) {
  if (id == HQ)
    return s.sum[NGRAM + NCROSS + NY + p] * 4096;
  auto op = ops[id];
  if (op.kind)
    return s.sum[NGRAM + NCROSS + NY + NHQ + (op.kind - 1) * 9 + p] * 4096;
  I result = s.sum[NGRAM + NCROSS + p] * 4096;
  int idx[3] = {op.a, op.b, op.c}, w[3] = {op.wa, op.wb, op.wc};
  for (int a = 0; a < 3; ++a) {
    if (!w[a] || idx[a] == 4)
      continue;
    int i = feature_index(idx[a]);
    result -= I(128) * w[a] * s.sum[NGRAM + p * 8 + i];
    for (int b = 0; b < 3; ++b) {
      if (!w[b] || idx[b] == 4)
        continue;
      int j = feature_index(idx[b]);
      result += I(w[a]) * w[b] * s.sum[gram_index(i, j)];
    }
  }
  return std::max<I>(0, result);
}
bool allowed(int p, int id, int bank, const std::vector<Op> &ops) {
  if (id == HQ || ops[id].kind || bank == 2)
    return true;
  auto op = ops[id];
  int dy = p / 3 - 1, dx = p % 3 - 1;
  auto good = [&](int i, int weight) {
    if (!weight || i == 4)
      return true;
    int y = cy[i], x = cx[i];
    return (dy == 0 ? y == 0 : (y == 0 || y == dy)) &&
           (dx == 0 ? x == 0 : (x == 0 || x == dx));
  };
  return good(op.a, op.wa) && good(op.b, op.wb) && good(op.c, op.wc);
}
Block fit(const Stats &s, const std::vector<Op> &ops,
          const Block *parent = nullptr, int bank = 0) {
  Block b;
  b.op.fill(HQ);
  if (s.n < (parent ? 24 : 48))
    return parent ? *parent : b;
  for (int p = 0; p < 9; ++p) {
    int best = parent ? parent->op[p] : 4;
    I bestcost = cost(s, p, best, ops);
    auto consider = [&](int id) {
      if (!allowed(p, id, bank, ops))
        return;
      I loss = cost(s, p, id, ops); // A tiny fixed complexity margin avoids
                                    // chasing sub-DN differences.
      if (loss + I(s.n) * 64 < bestcost) {
        bestcost = loss;
        best = id;
      }
    };
    consider(HQ);
    for (size_t i = 0; i < ops.size(); ++i)
      consider(int(i));
    b.op[p] = uint8_t(best);
  }
  return b;
}
long double block_cost(const Stats &s, const Block &b,
                       const std::vector<Op> &ops) {
  long double loss = 0;
  for (int p = 0; p < 9; ++p)
    loss += cost(s, p, b.op[p], ops);
  return loss / 4096;
}
void candidates(Stats &s, const std::vector<Op> &ops, const Block *parent,
                int bank) {
  for (int p = 0; p < 9; ++p) {
    std::vector<int> ids{HQ, 4};
    if (parent && parent->op[p] != HQ && parent->op[p] != 4)
      ids.push_back(parent->op[p]);
    std::vector<int> order;
    for (size_t i = 0; i < ops.size(); ++i)
      if (allowed(p, int(i), bank, ops))
        order.push_back(int(i));
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      I ca = cost(s, p, a, ops), cb = cost(s, p, b, ops);
      return ca != cb ? ca < cb : ops[a].reserved > ops[b].reserved;
    });
    for (int i : order)
      if (std::find(ids.begin(), ids.end(), i) == ids.end()) {
        ids.push_back(i);
        if (ids.size() == K)
          break;
      }
    while (ids.size() < K)
      ids.push_back(4);
    for (int k = 0; k < K; ++k)
      s.candidates[p * K + k] = uint8_t(ids[k]);
  }
}
Block exact_fit(const Stats &s, const Block *parent) {
  Block b;
  b.op.fill(HQ);
  if (s.n < (parent ? 24 : 48))
    return parent ? *parent : b;
  for (int p = 0; p < 9; ++p) {
    int best = 0;
    for (int k = 1; k < K; ++k) {
      uint64_t a = s.exact[p * K + k], bb = s.exact[p * K + best];
      if (a < bb || (a == bb && s.candidates[p * K + best] == HQ &&
                     s.candidates[p * K + k] != HQ))
        best = k;
    }
    b.op[p] = s.candidates[p * K + best];
  }
  return b;
}
