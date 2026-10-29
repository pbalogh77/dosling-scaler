// Offline 3x3 primary / full 5x5 exception fitting. No test labels are read.
#include "fit_common.hpp"
static void transform25(uint32_t *c, uint32_t *target, int code) {
  uint32_t out[25], t[9];
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x) {
      int yy = y, xx = (code & 4) ? -x : x;
      for (int r = 0; r < (code & 3); ++r) {
        int q = yy;
        yy = xx;
        xx = -q;
      }
      out[(yy + 2) * 5 + xx + 2] = c[(y + 2) * 5 + x + 2];
      if (std::abs(x) <= 1 && std::abs(y) <= 1)
        t[(yy + 1) * 3 + xx + 1] = target[(y + 1) * 3 + x + 1];
    }
  std::copy_n(out, 25, c);
  std::copy_n(t, 9, target);
}
template <class Fn>
void iterate5(const std::string &list, bool augment, Fn fn) {
  std::ifstream lines(list);
  if (!lines)
    throw std::runtime_error("Missing dataset list");
  std::string path;
  int frames = 0;
  while (std::getline(lines, path)) {
    if (path.empty())
      continue;
    std::ifstream file(path, std::ios::binary);
    uint32_t w = 0, h = 0, nr = 0, seed = 0;
    file.read((char *)&w, 4);
    file.read((char *)&h, 4);
    file.read((char *)&nr, 4);
    file.read((char *)&seed, 4);
    if (!file || !w || !h || w > 4096 || h > 4096 || nr > 65536)
      throw std::runtime_error("Invalid frame");
    std::vector<uint8_t> src(size_t(w) * h * 3), target(size_t(w) * h * 27);
    std::vector<dl_rect> regions(nr);
    file.read((char *)src.data(), src.size());
    file.read((char *)target.data(), target.size());
    file.read((char *)regions.data(), regions.size() * sizeof(dl_rect));
    if (!file)
      throw std::runtime_error("Truncated frame");
    for (auto r : regions)
      if (r.x0 < 0 || r.y0 < 0 || r.x1 > int(w) || r.y1 > int(h) ||
          r.x0 >= r.x1 || r.y0 >= r.y1)
        throw std::runtime_error("Invalid regions");
    Frame f;
    f.prepare(src.data(), int(w), int(h), w * 3, regions.data(), nr, true);
    prepare_ufo(f);
    for (int y = 0; y < int(h); ++y)
      for (int x = 0; x < int(w); ++x) {
        size_t p = size_t(y) * w + x;
        uint32_t hash = hash32(uint32_t(p) ^ seed);
        if (hash % (w * h) >= (f.text[p] ? 16000u : 5000u) || f.ufo_rules[p])
          continue;
        uint32_t patch[25], t[9], c[9];
        f.core(x, y, c);
        int bank = f.bank(x, y, c).index;
        size_t center = size_t(y + 2) * f.pitch + x + 2;
        for (int yy = -2; yy <= 2; ++yy)
          for (int xx = -2; xx <= 2; ++xx)
            patch[(yy + 2) * 5 + xx + 2] = f.pixels[center + yy * f.pitch + xx];
        for (int yy = 0; yy < 3; ++yy)
          for (int xx = 0; xx < 3; ++xx) {
            const auto q =
                &target[(size_t(y * 3 + yy) * w * 3 + x * 3 + xx) * 3];
            t[yy * 3 + xx] =
                (uint32_t(q[0]) << 16) | (uint32_t(q[1]) << 8) | q[2];
          }
        if (augment)
          transform25(patch, t, int(hash >> 29));
        for (int yy = -1; yy <= 1; ++yy)
          for (int xx = -1; xx <= 1; ++xx)
            c[(yy + 1) * 3 + xx + 1] = patch[(yy + 2) * 5 + xx + 2];
        uint32_t far = 0;
        for (int i = 0; i < 9; ++i)
          far = std::max(far, uint32_t(distance(c[i], c[4])));
        uint32_t threshold = std::max(48u, far >> 5), key = 0;
        int bit = 0;
        for (int i = 0; i < 9; ++i)
          if (i != 4) {
            if (uint32_t(distance(c[i], c[4])) > threshold)
              key |= 1u << bit;
            ++bit;
          }
        uint32_t outer = 0;
        bit = 0;
        for (int yy = -2; yy <= 2; ++yy)
          for (int xx = -2; xx <= 2; ++xx)
            if (std::abs(yy) == 2 || std::abs(xx) == 2) {
              if (uint32_t(distance(patch[(yy + 2) * 5 + xx + 2], c[4])) >
                  threshold)
                outer |= 1u << bit;
              ++bit;
            }
        fn(uint32_t(bank * 256 + key), uint16_t(outer), c, t);
      }
    if (++frames % 50 == 0)
      std::cerr << frames << " frames scanned\n";
  }
}
int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::cerr
          << "Usage: lut-fit5 train.txt validation.txt model.bin report.json\n";
      return 2;
    }
    if (std::filesystem::exists(argv[3]) || std::filesystem::exists(argv[4]) ||
        std::filesystem::absolute(argv[3]).lexically_normal() ==
            std::filesystem::absolute(argv[4]).lexically_normal())
      throw std::runtime_error("Use distinct new output paths");
    Model m;
    m.version = 4;
    m.ops = operations();
    std::unordered_map<uint32_t, uint32_t> counts;
    uint64_t samples = 0;
    std::cerr << "Counting full 5x5 states after hardcoded rules\n";
    iterate5(
        argv[1], true,
        [&](uint32_t prefix, uint16_t key, const uint32_t *, const uint32_t *) {
          ++counts[(prefix << 16) | key];
          ++samples;
        });
    std::unordered_map<uint32_t, Stats> states;
    std::vector<uint32_t> keys;
    for (auto item : counts)
      if (item.second >= 24)
        keys.push_back(item.first);
    std::sort(keys.begin(), keys.end());
    for (auto key : keys)
      states.try_emplace(key);
    std::array<Stats, BANKS * 256> base_stats;
    std::cerr << "Accumulating integer statistics: " << keys.size()
              << " supported exception states\n";
    iterate5(argv[1], true,
             [&](uint32_t prefix, uint16_t key, const uint32_t *c,
                 const uint32_t *t) {
               auto v = contribution(c, t);
               base_stats[prefix].add(v);
               auto it = states.find((prefix << 16) | key);
               if (it != states.end())
                 it->second.add(v);
             });
    std::array<Block, BANKS * 256> base;
    std::map<uint32_t, Block> ext;
    for (int prefix = 0; prefix < int(BANKS * 256); ++prefix) {
      base[prefix] = fit(base_stats[prefix], m.ops, nullptr, prefix / 256);
      candidates(base_stats[prefix], m.ops, nullptr, prefix / 256);
    }
    for (auto key : keys) {
      uint32_t prefix = key >> 16;
      auto &st = states.at(key);
      ext[key] = fit(st, m.ops, &base[prefix], int(prefix / 256));
      candidates(st, m.ops, &base[prefix], int(prefix / 256));
    }
    std::cerr << "Scoring exact integer candidates on training only\n";
    iterate5(argv[1], true,
             [&](uint32_t prefix, uint16_t key, const uint32_t *c,
                 const uint32_t *t) {
               auto score = [&](Stats &st) {
                 for (int k = 0; k < K; ++k) {
                   Block b;
                   for (int p = 0; p < 9; ++p)
                     b.op[p] = st.candidates[p * K + k];
                   uint32_t out[9];
                   evaluate(m, b, c, out);
                   for (int p = 0; p < 9; ++p)
                     st.exact[p * K + k] += distance(out[p], t[p]);
                 }
               };
               score(base_stats[prefix]);
               auto it = states.find((prefix << 16) | key);
               if (it != states.end())
                 score(it->second);
             });
    for (int prefix = 0; prefix < int(BANKS * 256); ++prefix)
      base[prefix] = exact_fit(base_stats[prefix], nullptr);
    for (auto key : keys)
      ext[key] = exact_fit(states.at(key), &base[key >> 16]);
    std::map<Block, uint16_t> ids;
    auto id = [&](const Block &b) {
      auto it = ids.find(b);
      if (it != ids.end())
        return it->second;
      if (m.blocks.size() >= 65535)
        throw std::runtime_error("Recipe count overflow");
      auto n = uint16_t(m.blocks.size());
      ids.emplace(b, n);
      m.blocks.push_back(b);
      return n;
    };
    std::array<Exception5, BANKS * 256> proposed;
    for (uint32_t prefix = 0; prefix < BANKS * 256; ++prefix)
      m.primary[prefix] = {id(base[prefix]), NO_EXT};
    for (const auto &item : ext) {
      uint32_t prefix = item.first >> 16;
      if (item.second.op != base[prefix].op)
        proposed[prefix].leaves.push_back(
            {uint16_t(item.first), id(item.second)});
    }
    for (auto &table : proposed)
      table.compress_masks();
    struct Loss {
      uint64_t n = 0, base = 0, candidate = 0;
    };
    std::array<Loss, BANKS * 256> val;
    uint64_t validation_samples = 0;
    std::cerr << "Selecting exception flags on separate validation games\n";
    iterate5(argv[2], false,
             [&](uint32_t prefix, uint16_t key, const uint32_t *c,
                 const uint32_t *t) {
               auto &v = val[prefix];
               ++v.n;
               ++validation_samples;
               uint32_t out[9];
               evaluate(m, base[prefix], c, out);
               uint64_t baseline = 0;
               for (int p = 0; p < 9; ++p)
                 baseline += distance(out[p], t[p]);
               v.base += baseline;
               auto chosen = proposed[prefix].lookup(key);
               if (chosen == NO_EXT) {
                 v.candidate += baseline;
                 return;
               }
               evaluate(m, m.blocks[chosen], c, out);
               for (int p = 0; p < 9; ++p)
                 v.candidate += distance(out[p], t[p]);
             });
    uint64_t baseline = 0, selected_loss = 0, exact = 0, masked = 0;
    std::array<int, 3> flags{};
    for (uint32_t prefix = 0; prefix < BANKS * 256; ++prefix) {
      auto v = val[prefix];
      baseline += v.base;
      if (v.n >= 96 && v.candidate * 100 < v.base * 99 &&
          (v.base - v.candidate) * 100 > v.n * 81) {
        m.primary[prefix].extended = uint16_t(m.exceptions.size());
        m.exceptions.push_back(std::move(proposed[prefix]));
        auto &e = m.exceptions.back();
        exact += e.leaves.size();
        masked += e.masked.size();
        ++flags[prefix / 256];
        selected_loss += v.candidate;
      } else
        selected_loss += v.base;
    }
    // Only recipes referenced by selected tables survive in the shipped model.
    std::vector<Block> compact;
    std::map<uint16_t, uint16_t> remap;
    auto keep = [&](uint16_t old) {
      auto it = remap.find(old);
      if (it != remap.end())
        return it->second;
      auto n = uint16_t(compact.size());
      compact.push_back(m.blocks[old]);
      remap.emplace(old, n);
      return n;
    };
    for (auto &e : m.primary)
      e.recipe = keep(e.recipe);
    for (auto &e : m.exceptions) {
      for (auto &l : e.leaves)
        l.recipe = keep(l.recipe);
      for (auto &r : e.masked)
        r.recipe = keep(r.recipe);
    }
    m.blocks = std::move(compact);
    save(m, argv[3]);
    auto loaded = load(argv[3]);
    for (size_t i = 0; i < m.exceptions.size(); ++i)
      for (int key = 0; key < 65536; ++key)
        if (m.exceptions[i].lookup(uint16_t(key)) !=
            loaded.exceptions[i].lookup(uint16_t(key)))
          throw std::runtime_error("Serialized mask lookup mismatch");
    auto frame_count = [](const char *path) {
      std::ifstream f(path);
      std::string s;
      uint64_t n = 0;
      while (std::getline(f, s))
        n += !s.empty();
      return n;
    };
    std::ofstream report(argv[4]);
    report << "{\n\"scope\":\"Offline finite-recipe fitting after hardcoded "
              "text rules; 3x3 primary flags and full 5x5 masked exceptions. "
              "No test labels or human preferences.\",\n\"training_frames\":"
           << frame_count(argv[1])
           << ",\n\"validation_frames\":" << frame_count(argv[2])
           << ",\n\"training_samples\":" << samples
           << ",\n\"validation_samples\":" << validation_samples
           << ",\n\"observed_5x5_states\":" << counts.size()
           << ",\n\"supported_training_states\":" << keys.size()
           << ",\n\"recipes\":" << m.blocks.size()
           << ",\n\"exception_prefixes\":" << m.exceptions.size()
           << ",\n\"exception_prefixes_by_bank\":[" << flags[0] << ","
           << flags[1] << "," << flags[2] << "],\n\"exact_rules\":" << exact
           << ",\n\"masked_rules\":" << masked
           << ",\n\"model_bytes\":" << model_bytes(m)
           << ",\n\"validation_base_mse\":"
           << double(baseline) / double(validation_samples * 27)
           << ",\n\"validation_exception_mse\":"
           << double(selected_loss) / double(validation_samples * 27)
           << ",\n\"mask_compression\":\"Exact union of same-recipe cubes; no "
              "added key coverage. Highest care-bit specificity first; maximum "
              "64 masked rules per prefix.\"\n}\n";
    std::cerr << m.exceptions.size() << " exception flags, " << masked
              << " masked rules, " << model_bytes(m) << " bytes\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
