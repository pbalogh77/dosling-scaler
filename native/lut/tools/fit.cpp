#include "fit_common.hpp"
int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::cerr
          << "Usage: lut-fit train.txt validation.txt model.bin report.json\n";
      return 2;
    }
    if (std::filesystem::exists(argv[3]) || std::filesystem::exists(argv[4]) ||
        std::filesystem::absolute(argv[3]).lexically_normal() ==
            std::filesystem::absolute(argv[4]).lexically_normal())
      throw std::runtime_error("Use distinct, new model and report paths; "
                               "prior outputs are preserved");
    Dataset train, val;
    std::cerr << "Accumulating training\n";
    accumulate(train, argv[1], true);
    std::cerr << "Accumulating validation (no fit)\n";
    accumulate(val, argv[2], false);
    Model m;
    m.ops = operations();
    std::array<Block, BANKS * 256> base;
    std::unordered_map<uint32_t, Block> ext;
    for (size_t p = 0; p < base.size(); ++p) {
      base[p] = fit(train.base[p], m.ops, nullptr, int(p / 256));
      candidates(train.base[p], m.ops, nullptr, int(p / 256));
    }
    for (auto &item : train.extended)
      candidates(item.second, m.ops, &base[(item.first / 256) % (BANKS * 256)],
                 int((item.first / 65536) % BANKS));
    std::cerr << "Exact integer candidate scoring on training only\n";
    iterate(argv[1], true,
            [&](uint32_t prefix, uint32_t outer, uint32_t relations,
                const uint32_t *c, const uint32_t *t) {
              auto &bs = train.base[prefix];
              auto &es = train.extended[prefix * 256 + outer];
              auto &rs =
                  train.extended[(BANKS * 256 + prefix) * 256 + relations];
              uint32_t cache[256][9];
              bool ready[256] = {};
              auto score = [&](Stats &s) {
                for (int p = 0; p < 9; ++p)
                  for (int k = 0; k < K; ++k) {
                    int id = s.candidates[p * K + k];
                    if (!ready[id]) {
                      Block b;
                      b.op.fill(uint8_t(id));
                      evaluate(m, b, c, cache[id]);
                      ready[id] = true;
                    }
                    s.exact[p * K + k] += distance(cache[id][p], t[p]);
                  }
              };
              score(bs);
              if (es.n >= 24)
                score(es);
              if (rs.n >= 24)
                score(rs);
            });
    for (size_t p = 0; p < base.size(); ++p)
      base[p] = exact_fit(train.base[p], nullptr);
    for (auto &item : train.extended)
      ext[item.first] =
          exact_fit(item.second, &base[(item.first / 256) % (BANKS * 256)]);
    std::cerr << "Exact validation gate selection (no coefficient fitting)\n";
    iterate(argv[2], false,
            [&](uint32_t prefix, uint32_t outer, uint32_t relations,
                const uint32_t *c, const uint32_t *t) {
              uint32_t a[9], b[9];
              evaluate(m, base[prefix], c, a);
              for (int plan = 0; plan < 2; ++plan) {
                uint32_t key = (plan * BANKS * 256 + prefix) * 256 +
                               (plan ? relations : outer);
                auto &st = val.extended[key];
                auto e = ext.find(key);
                if (e == ext.end())
                  std::copy_n(a, 9, b);
                else
                  evaluate(m, e->second, c, b);
                for (int p = 0; p < 9; ++p) {
                  st.validation_base += distance(a[p], t[p]);
                  st.validation_extended += distance(b[p], t[p]);
                }
              }
            });
    std::map<Block, uint16_t> ids;
    auto id = [&](const Block &b) {
      auto found = ids.find(b);
      if (found != ids.end())
        return found->second;
      if (m.blocks.size() >= 65535)
        throw std::runtime_error("Too many recipes");
      auto index = uint16_t(m.blocks.size());
      m.blocks.push_back(b);
      ids[b] = index;
      return index;
    };
    long double base_val = 0, extended_val = 0;
    uint64_t gates = 0;
    uint64_t relation_gates = 0;
    for (uint32_t prefix = 0; prefix < BANKS * 256; ++prefix) {
      m.primary[prefix] = {id(base[prefix]), NO_EXT};
      uint64_t baseline = 0, best = 0, count = 0;
      int selected = -1;
      for (int plan = 0; plan < 2; ++plan) {
        uint64_t loss = 0, current_base = 0, n = 0;
        for (int outer = 0; outer < 256; ++outer) {
          auto v =
              val.extended.find((plan * BANKS * 256 + prefix) * 256 + outer);
          if (v == val.extended.end())
            continue;
          current_base += v->second.validation_base;
          loss += v->second.validation_extended;
          n += v->second.n;
        }
        if (plan == 0) {
          baseline = current_base;
          best = baseline;
          count = n;
        }
        if (n >= 96 && loss * 100 < baseline * 99 &&
            (baseline - loss) * 100 > n * 81 && loss < best) {
          best = loss;
          selected = plan;
        }
      }
      base_val += baseline;
      extended_val += best;
      if (selected >= 0) {
        std::array<uint16_t, 256> table;
        table.fill(NO_EXT);
        for (int outer = 0; outer < 256; ++outer) {
          auto e = ext.find((selected * BANKS * 256 + prefix) * 256 + outer);
          if (e != ext.end() && e->second.op != base[prefix].op)
            table[outer] = id(e->second);
        }
        m.primary[prefix].extended =
            uint16_t(m.extended.size() | (selected << 15));
        m.extended.push_back(table);
        ++gates;
        relation_gates += selected == 1;
      }
    }
    save(m, argv[3]);
    std::ifstream bin(argv[3], std::ios::binary | std::ios::ate);
    std::ofstream report(argv[4]);
    report << "{\n \"scope\": \"Offline finite-recipe distillation; exact "
              "integer recipe scoring on training, extension gates selected on "
              "game-held-out validation, no test labels used\",\n "
              "\"training_frames\": "
           << train.frames << ",\n \"training_samples\": " << train.samples
           << ",\n \"validation_frames\": " << val.frames
           << ",\n \"validation_samples\": " << val.samples
           << ",\n \"extended_states_observed\": " << train.extended.size()
           << ",\n \"operators\": " << m.ops.size()
           << ",\n \"recipes\": " << m.blocks.size()
           << ",\n \"extended_prefixes\": " << gates
           << ",\n \"model_bytes\": " << bin.tellg()
           << ",\n \"validation_base_mse\": "
           << double(base_val / (val.samples * 27))
           << ",\n \"validation_extended_mse\": "
           << double(extended_val / (val.samples * 27)) << "\n}\n";
    std::cerr << m.blocks.size() << " recipes, " << relation_gates
              << " relation gates, " << gates << " extended prefixes, "
              << bin.tellg() << " bytes\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
