#include "model.hpp"
#include <filesystem>
#include <iostream>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x))                                                                  \
      throw std::runtime_error(#x);                                            \
  } while (0)
using namespace lut;
static Frame frame(const std::vector<uint8_t> &s, int w, int h,
                   bool text = true) {
  Frame f;
  dl_rect r{0, 0, w, h};
  f.prepare(s.data(), w, h, w * 3, text ? &r : nullptr, text ? 1 : 0, true);
  return f;
}
static void put(std::vector<uint8_t> &s, int w, int x, int y, uint8_t v) {
  for (int k = 0; k < 3; ++k)
    s[(y * w + x) * 3 + k] = v;
}
int main() {
  try {
    // Check sparse/dense index boundaries and the full 65,536-leaf offset.
    // A flat oracle keeps this independent of the bucket implementation.
    for (unsigned count : {0u, 1u, 60u, 256u, 257u, 65536u}) {
      Exception5 indexed;
      std::array<uint16_t, 65536> expected;
      expected.fill(NO_EXT);
      for (unsigned i = 0; i < count; ++i) {
        uint16_t key = uint16_t(i * 40503u), recipe = uint16_t(i % 3);
        indexed.leaves.push_back({key, recipe});
        expected[key] = recipe;
      }
      indexed.index();
      for (unsigned key = 0; key < 65536; ++key)
        CHECK(indexed.lookup(uint16_t(key)) == expected[key]);
    }
    Exception5 table;
    table.leaves = {
        {0x0010, 1}, {0x0011, 1}, {0x0020, 2}, {0x0021, 2}, {0x0022, 3}};
    table.index();
    std::array<uint16_t, 65536> before;
    for (int i = 0; i < 65536; ++i)
      before[i] = table.lookup(uint16_t(i));
    table.compress_masks();
    CHECK(!table.masked.empty());
    for (int i = 0; i < 65536; ++i)
      CHECK(table.lookup(uint16_t(i)) == before[i]);
    Exception5 priority;
    priority.leaves = {{0x1234, 6}};
    priority.masked = {{0x1000, 0xf000, 4}, {0x1200, 0xff00, 5}};
    priority.index();
    CHECK(priority.lookup(0x1234) == 6);
    CHECK(priority.lookup(0x1256) == 5);
    CHECK(priority.lookup(0x1fff) == 4);
    CHECK(priority.lookup(0x2000) == NO_EXT);
    // Every one of the sixteen outer-ring positions participates, including the
    // eight positions absent from the old 17-pixel footprint.
    int bit = 0;
    for (int y = -2; y <= 2; ++y)
      for (int x = -2; x <= 2; ++x)
        if (std::abs(y) == 2 || std::abs(x) == 2) {
          std::vector<uint8_t> s(7 * 7 * 3);
          put(s, 7, 3 + x, 3 + y, 255);
          auto f = frame(s, 7, 7, false);
          CHECK(f.keys[3 * 7 + 3] == 0);
          CHECK(f.outer5(3, 3, 48) == uint16_t(1u << bit++));
        }
    CHECK(bit == 16);
    for (int sy : {-1, 1})
      for (int sx : {-1, 1})
        for (int alpha : {64, 128, 192}) {
          std::vector<uint8_t> s(11 * 11 * 3);
          for (int y = 0; y < 11; ++y)
            for (int x = 0; x < 11; ++x)
              put(s, 11, x, y, (x - 5) * sx > 0 || (y - 5) * sy > 0 ? 255 : 0);
          put(s, 11, 5, 5, uint8_t(alpha));
          auto f = frame(s, 11, 11);
          prepare_ufo(f);
          uint32_t c[9], out[9];
          f.core(5, 5, c);
          CHECK(ufo_block(f, 5, 5, c, out) == 1);
          int sum = 0;
          for (auto p : out) {
            auto z = rgb(p);
            CHECK(z.r == z.g && z.g == z.b);
            sum += z.r;
          }
          CHECK(std::abs(sum - alpha * 9) <= 18);
          auto no_text = frame(s, 11, 11, false);
          prepare_ufo(no_text);
          // A fully supported two-surface AA cut is now general geometry,
          // without text recognition. It must preserve the source average.
          uint32_t no_core[9],no_out[9];no_text.core(5,5,no_core);
          CHECK(ufo_block(no_text,5,5,no_core,no_out)==1);
          int no_sum=0;for(auto value:no_out)no_sum+=rgb(value).r;
          CHECK(std::abs(no_sum-alpha*9)<=18);
        }
    // Near-solid source mixtures imply a small clipped tip. Permit the
    // existing AA primitive only after the hardcoded 5x5 endpoint admission;
    // the learned AA operators must keep their original range.
    for (int sy : {-1, 1})
      for (int sx : {-1, 1})
        for (bool invert : {false, true})
          for (int alpha : {222, 230}) {
            std::vector<uint8_t> s(11 * 11 * 3);
            for (int y = 0; y < 11; ++y)
              for (int x = 0; x < 11; ++x) {
                int value = (x - 5) * sx > 0 || (y - 5) * sy > 0 ? 255 : 0;
                put(s, 11, x, y, uint8_t(invert ? 255 - value : value));
              }
            put(s, 11, 5, 5, uint8_t(invert ? 255 - alpha : alpha));
            auto f = frame(s, 11, 11);
            prepare_ufo(f);
            uint32_t c[9], out[9];
            f.core(5, 5, c);
            CHECK(ufo_block(f, 5, 5, c, out) == 1);
            int kind = (sy > 0 ? 0 : 2) + (sx > 0 ? 1 : 2);
            CHECK(!aa_block(kind, c, out));
            CHECK(aa_block(kind, c, out, true));
            int sum = 0;
            for (auto value : out)
              sum += rgb(value).r;
            CHECK(std::abs(sum - (invert ? 255 - alpha : alpha) * 9) <= 18);
            // Six tonal samples still lie on the same gray segment, but
            // undermine a two-surface corner. Keep the core/solid anchor.
            for (auto d : std::array<std::array<int, 2>, 6>{
                     {{{-2, -2}}, {{2, -2}}, {{-2, 2}}, {{2, 2}},
                      {{-1, -2}}, {{1, 2}}}})
              put(s, 11, 5 + d[0], 5 + d[1], uint8_t(invert ? 127 : 128));
            auto textured = frame(s, 11, 11);
            prepare_ufo(textured);
            int rule = textured.ufo_rules[5 * 11 + 5];
            CHECK(rule < 1 || rule > 4);
          }
    std::vector<uint8_t> t(11 * 11 * 3);
    for (int x = 3; x <= 7; ++x)
      put(t, 11, x, 5, 255);
    for (int y = 5; y <= 7; ++y)
      put(t, 11, 5, y, 255);
    auto tf = frame(t, 11, 11);
    prepare_ufo(tf);
    uint32_t core[9], out[9];
    tf.core(5, 5, core);
    CHECK(ufo_block(tf, 5, 5, core, out) == 2);
    for (auto v : out)
      CHECK(v == 0xffffff);
    // Mixed one/two-pixel stroke widths need square inner T gaps too. Full
    // 5x5 source rectangles distinguish them from bent or missing arms.
    for (int rotation = 0; rotation < 4; ++rotation)
      for (bool invert : {false, true})
        for (auto widths : std::array<std::array<int, 2>, 3>{
                 {{{1, 2}}, {{2, 1}}, {{2, 2}}}})
          for (int defect : {0, 1, 2}) {
            const int bar = widths[0], stem = widths[1];
            std::vector<uint8_t> s(15 * 15 * 3, uint8_t(invert ? 223 : 32));
            auto map = [&](int x, int y) {
              for (int r = 0; r < rotation; ++r) {
                int next = -y;
                y = x;
                x = next;
              }
              return std::array<int, 2>{x + 7, y + 7};
            };
            auto paint = [&](int x, int y, int value) {
              auto at = map(x, y);
              put(s, 15, at[0], at[1], uint8_t(invert ? 255 - value : value));
            };
            for (int y = -4; y <= 4; ++y)
              for (int x = -4; x <= 4; ++x) {
                bool on = (y >= 0 && y < bar) || (y < 0 && x >= 0 && x < stem);
                if (defect == 1 && x < 0 && y >= 0 && y < bar)
                  on = false; // Remove one complete bar arm: this is an L.
                paint(x, y, on ? 224 : 32);
              }
            if (defect == 2)
              for (int x = -4; x <= 4; ++x)
                paint(x, 2, 128); // A third tonal layer under the junction.
            auto f = frame(s, 15, 15);
            prepare_ufo(f);
            for (int x : {-1, stem}) {
              auto at = map(x, -1);
              uint32_t c[9], out[9];
              f.core(at[0], at[1], c);
              int result = ufo_block(f, at[0], at[1], c, out);
              if (!defect) {
                CHECK(result == 2);
                for (auto value : out)
                  CHECK(value == (invert ? 0xdfdfdfu : 0x202020u));
              } else
                CHECK(result != 2);
            }
          }
    // A short F-like arm has a flat endpoint; an AA/bent continuation at a
    // front corner must keep its original rendering instead of forcing a cap.
    for (int rotation = 0; rotation < 4; ++rotation)
      for (bool invert : {false, true})
        for (int defect : {0, 1, 2}) {
          std::vector<uint8_t> s(15 * 15 * 3, uint8_t(invert ? 223 : 32));
          auto map = [&](int x, int y) {
            for (int r = 0; r < rotation; ++r) {
              int next = -y;
              y = x;
              x = next;
            }
            return std::array<int, 2>{x + 7, y + 7};
          };
          auto paint = [&](int x, int y, int value) {
            auto at = map(x, y);
            put(s, 15, at[0], at[1], uint8_t(invert ? 255 - value : value));
          };
          for (int y = -4; y <= 4; ++y)
            for (int x = -4; x <= 4; ++x)
              paint(x, y, ((y >= 0 && y < 2) ||
                           (y >= -2 && y < 0 && x == 0)) ? 224 : 32);
          if (defect)
            paint(1, -3, defect == 1 ? 224 : 128);
          auto f = frame(s, 15, 15);
          prepare_ufo(f);
          auto at = map(0, -2);
          uint32_t c[9], out[9];
          f.core(at[0], at[1], c);
          int result = ufo_block(f, at[0], at[1], c, out);
          if (!defect) {
            CHECK(result == 2);
            for (auto value : out)
              CHECK(value == (invert ? 0x1f1f1fu : 0xe0e0e0u));
          } else
            CHECK(result != 2);
        }
    // A real shaded-font E junction: brightness changes down its stem while
    // the three source arms remain orthogonal. A bent continuation must not
    // be squared off. Rotate and invert the source to cover both polarities.
    for (int rotation = 0; rotation < 4; ++rotation)
      for (bool invert : {false, true})
        for (bool bent : {false, true}) {
          std::vector<uint8_t> shaded(11 * 11 * 3,
                                      uint8_t(invert ? 183 : 72));
          auto p = [&](int x, int y, int value) {
            for (int r = 0; r < rotation; ++r) {
              int next = -y;
              y = x;
              x = next;
            }
            put(shaded, 11, x + 5, y + 5,
                uint8_t(invert ? 255 - value : value));
          };
          p(-1, -1, 80);
          p(-1, 1, 80);
          p(1, -1, 36);
          p(1, 1, 36);
          p(-1, 0, 60);
          p(0, -2, bent ? 72 : 129);
          p(0, -1, 145);
          p(0, 0, 157);
          p(0, 1, 170);
          p(0, 2, 182);
          p(1, 0, 157);
          if (bent)
            p(-1, -2, 157);
          auto sf = frame(shaded, 11, 11);
          prepare_ufo(sf);
          sf.core(5, 5, core);
          CHECK(ufo_block(sf, 5, 5, core, out) == (bent ? 0 : 2));
          if (!bent)
            for (auto v : out)
              CHECK(v == (invert ? 0x626262u : 0x9d9d9du));
        }
    // Narrow antialiased S bends between other letters: the two inner
    // corners need their common continuation, rather than unrelated glyphs.
    for (int rotation = 0; rotation < 4; ++rotation)
      for (bool invert : {false, true})
        for (int alpha : {64, 128, 192})
          for (int defect : {0, 1, 2}) {
            std::vector<uint8_t> s(25 * 25 * 3,
                                    uint8_t(invert ? 235 : 20));
            auto location = [&](int x, int y) {
              for (int r = 0; r < rotation; ++r) {
                int next = y;
                y = 24 - x;
                x = next;
              }
              return std::array<int, 2>{x, y};
            };
            auto paint = [&](int x, int y, int value) {
              auto p = location(x, y);
              put(s, 25, p[0], p[1], uint8_t(invert ? 255 - value : value));
            };
            const std::string a[] = {"+#+", "#.#", "###", "#.#", "#.#"},
                              bend[] = {"+##", "#..", "+#+", "..#", "##+"},
                              e[] = {"##", "#.", "##", "#.", "##"};
            int mixture = 20 + (180 * alpha + 128) / 256;
            for (int y = 0; y < 5; ++y) {
              auto row = a[y] + "." + bend[y] + "." + e[y] + "." + bend[y];
              for (int x = 0; x < int(row.size()); ++x)
                paint(x + 5, y + 9,
                      row[x] == '#' ? 200 : row[x] == '+' ? mixture : 20);
            }
            if (defect == 1)
              paint(10, 9, 20); // Missing outer stroke witness.
            if (defect == 2)
              paint(11, 11, 20 + (180 * (alpha < 128 ? 192 : 64) + 128) / 256);
            auto sf = frame(s, 25, 25);
            prepare_ufo(sf);
            for (int x : {9, 11}) {
              auto p = location(x, 11);
              sf.core(p[0], p[1], core);
              CHECK(ufo_block(sf, p[0], p[1], core, out) == (defect ? 0 : 1));
              if (!defect) {
                int sum = 0;
                for (auto v : out)
                  sum += rgb(v).r;
                CHECK(std::abs(sum - 9 * (invert ? 255 - mixture : mixture)) <= 18);
              }
            }
          }
    // The same source notch must keep its shape in light and dark text.
    for (auto colors : {std::array<int, 2>{255, 0}, {200, 40}})
      for (int alpha : {0, 64, 128, 192, 219}) {
        std::array<std::array<uint32_t, 27>, 2> rendered;
        for (int inverse = 0; inverse < 2; ++inverse) {
          int fg = inverse ? 255 - colors[0] : colors[0],
              bg = inverse ? 255 - colors[1] : colors[1];
          int mixture = colors[1] + ((colors[0] - colors[1]) * alpha + 128) / 256;
          if (inverse)
            mixture = 255 - mixture;
          std::vector<uint8_t> wedge(11 * 11 * 3, uint8_t(bg));
          for (auto xy : {std::array<int, 2>{3, 5}, {4, 5}, {5, 4}, {5, 6}})
            put(wedge, 11, xy[0], xy[1], uint8_t(fg));
          put(wedge, 11, 5, 5, uint8_t(mixture));
          auto wf = frame(wedge, 11, 11);
          prepare_ufo(wf);
          auto &rows = rendered[inverse];
          for (int y = 4; y <= 6; ++y) {
            wf.core(5, y, core);
            CHECK(ufo_block(wf, 5, y, core, rows.data() + (y - 4) * 9) == 3);
          }
          for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 3; ++x)
              CHECK(rows[y * 3 + x] == rows[(8 - y) * 3 + x]);
          // Breaking the stem continuation must prevent a forced wedge.
          put(wedge, 11, 3, 5, uint8_t(bg));
          auto broken = frame(wedge, 11, 11);
          prepare_ufo(broken);
          broken.core(5, 5, core);
          CHECK(ufo_block(broken, 5, 5, core, out) != 3);
        }
        for (int i = 0; i < 27; ++i)
          CHECK(std::abs(rgb(rendered[0][i]).r + rgb(rendered[1][i]).r - 255) <= 1);
      }
    // A gray foreground stroke above a black shadow on orange is two color
    // layers. Its shadow's local V shape must not be forced into a text notch.
    std::vector<uint8_t> layered(11 * 11 * 3);
    auto color = [&](int x, int y, std::array<uint8_t, 3> c) {
      for (int k = 0; k < 3; ++k)
        layered[(y * 11 + x) * 3 + k] = c[k];
    };
    for (int y = 0; y < 11; ++y)
      for (int x = 0; x < 11; ++x)
        color(x, y, {176, 84, 0});
    for (auto xy : {std::array<int, 2>{3, 5}, {4, 5}, {5, 4}, {5, 6}})
      color(xy[0], xy[1], {0, 0, 0});
    color(5, 5, {132, 63, 0});
    for (int x = 3; x <= 5; ++x)
      color(x, 3, {180, 180, 180});
    auto layers = frame(layered, 11, 11);
    prepare_ufo(layers);
    layers.core(5, 5, core);
    CHECK(ufo_block(layers, 5, 5, core, out) != 3);
    std::vector<uint8_t> checker(17 * 17 * 3);
    for (int y = 0; y < 17; ++y)
      for (int x = 0; x < 17; ++x)
        put(checker, 17, x, y, (x + y) % 2 ? 255 : 0);
    auto cf = frame(checker, 17, 17);
    prepare_ufo(cf);
    CHECK(std::none_of(cf.ufo_rules.begin(), cf.ufo_rules.end(),
                       [](auto x) { return x != 0; }));
    // Full C API: the exception flag dispatches only a matching outer-ring key;
    // an unseen key inherits its primary recipe. Serialized masks round-trip.
    Model model;
    model.version = 4;
    model.ops = operations();
    Block base, leaf;
    base.op.fill(4);
    leaf.op.fill(3);
    model.blocks = {base, leaf};
    for (auto &e : model.primary)
      e = {0, NO_EXT};
    std::vector<uint8_t> s(7 * 7 * 3);
    put(s, 7, 2, 3, 255);
    put(s, 7, 3, 1, 255);
    auto f = frame(s, 7, 7, false);
    uint16_t key = f.outer5(3, 3, f.threshold[24]);
    Exception5 ex;
    ex.leaves = {{key, 1}};
    ex.index();
    model.exceptions.push_back(ex);
    model.primary[f.keys[24]] = {0, 0};
    const auto path =
        std::filesystem::temp_directory_path() / "dos-lut-exception-unit.bin";
    save(model, path.c_str());
    auto loaded = load(path.c_str());
    CHECK(loaded.version == 4);
    CHECK(loaded.exceptions[0].lookup(key) == 1);
    auto context = dl_create(path.c_str());
    CHECK(context);
    std::vector<uint8_t> dst(7 * 7 * 27);
    dl_stats stats{};
    CHECK(dl_scale3(context, s.data(), 7, 7, 21, dst.data(), 63, nullptr, 0,
                    &stats) == 0);
    CHECK(dst[(9 * 21 + 9) * 3] == 255);
    dl_detail_stats details{};
    details.struct_size = sizeof(details);
    CHECK(dl_last_details(context, &details) == 0);
    CHECK(details.exception_hits > 0);
    CHECK(dl_model_bytes(context) == model_bytes(model));
    dl_destroy(context);
    std::filesystem::remove(path);
    std::cout << "Adaptive sparse/dense index, full 5x5 ring bits, exact mask compression, specificity, UFO "
                 "corner ratios/orientations, T joins, symmetric R wedge, "
                 "text-only admission and C ABI exception dispatch passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
