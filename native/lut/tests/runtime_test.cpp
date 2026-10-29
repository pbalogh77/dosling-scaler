#include "dos_lut.h"
#include "hqx.h"
#include "model.hpp"
#include <atomic>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x))                                                                  \
      throw std::runtime_error(#x);                                            \
  } while (0)
int main(int argc, char **argv) {
  try {
    CHECK(argc == 2);
    std::unique_ptr<dl_context, decltype(&dl_destroy)> context(
        dl_create(argv[1]), dl_destroy);
    CHECK(context);
    CHECK(dl_model_bytes(context.get()) > 0);
    CHECK(!dl_create("missing-model-no-file.bin"));
    std::mt19937 rng(93);
    // Prove the isolated fallback is actual HQ3x on source borders and
    // interiors.
    int w = 23, h = 17;
    std::vector<uint32_t> packed(w * h), hq(w * h * 9);
    for (auto &v : packed)
      v = rng() & 0xffffff;
    hq3x_32(packed.data(), hq.data(), w, h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        uint32_t core[9], out[9];
        for (int i = 0; i < 9; ++i)
          core[i] =
              packed[size_t(std::max(0, std::min(h - 1, y + lut::cy[i]))) * w +
                     std::max(0, std::min(w - 1, x + lut::cx[i]))];
        ds_hq3x_block(core, out);
        for (int oy = 0; oy < 3; ++oy)
          for (int ox = 0; ox < 3; ++ox)
            CHECK(out[oy * 3 + ox] ==
                  hq[size_t(y * 3 + oy) * w * 3 + x * 3 + ox]);
      }
    for (int height : {1, 2, 5, 17})
      for (int width : {1, 2, 3, 8, 19}) {
        size_t ss = width * 3 + 5, ds = width * 9 + 7;
        std::vector<uint8_t> s(ss * height, 0xa9), out(ds * height * 3, 0xcd),
            again(out.size(), 0xcd);
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width * 3; ++x)
            s[y * ss + x] = uint8_t(rng());
        auto original = s;
        dl_rect r{0, 0, width, height};
        CHECK(dl_scale3(context.get(), s.data(), width, height, ss, out.data(),
                        ds, &r, 1, nullptr) == 0);
        CHECK(dl_scale3(context.get(), s.data(), width, height, ss,
                        again.data(), ds, &r, 1, nullptr) == 0);
        CHECK(out == again);
        CHECK(s == original);
        for (int y = 0; y < height * 3; ++y)
          for (size_t x = width * 9; x < ds; ++x)
            CHECK(out[y * ds + x] == 0xcd);
        CHECK(dl_scale3(context.get(), s.data(), width, height, ss, s.data(),
                        ds, nullptr, 0, nullptr) == 1);
        CHECK(dl_scale3(context.get(), s.data(), width, height, ss, out.data(),
                        width * 9 - 1, nullptr, 0, nullptr) == 1);
        r.x0 = -1;
        CHECK(dl_scale3(context.get(), s.data(), width, height, ss, out.data(),
                        ds, &r, 1, nullptr) == 1);
      }
    // No new hue may appear outside the source convex hull. Neutral sources
    // must remain neutral for every method/bank, including antialias operators.
    for (int mode = 0; mode < 6; ++mode) {
      w = 31;
      h = 27;
      std::vector<uint8_t> src(w * h * 3), out(w * h * 27);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          uint8_t v = mode == 0 ? uint8_t(rng())
                      : mode == 4 && y < h / 2 ? 40
                      : mode == 5 && x + y < 13 ? 40
                      : mode >= 3 ? uint8_t((x + y) % 2 ? 40 : 100)
                                  : uint8_t((x + y) % 2 ? 0 : 255);
          for (int k = 0; k < 3; ++k)
            src[(y * w + x) * 3 + k] = v;
        }
      dl_rect r{0, 0, w, h};
      CHECK(dl_scale3(context.get(), src.data(), w, h, w * 3, out.data(), w * 9,
                      mode == 2 ? &r : nullptr, mode == 2 ? 1 : 0,
                      nullptr) == 0);
      for (size_t i = 0; i < out.size(); i += 3) {
        CHECK(out[i] == out[i + 1] && out[i] == out[i + 2]);
        if (mode >= 3)
          CHECK(out[i] >= 40 && out[i] <= 100);
      }
    }
    // A flat field between foreground and a one-pixel shadow is a distinct
    // third surface, even when its RGB happens to fit the AA mixture.
    if (context->impl.model.version >= 4) {
      constexpr int side = 9;
      std::vector<uint8_t> src(side * side * 3, 64), out(side * side * 27);
      bool ink[side][side]{};
      for (auto p : std::array<std::array<int,2>,7>{
               {{{2,2}},{{2,3}},{{2,4}},{{3,2}},{{3,4}},{{4,2}},{{4,3}}}})
        ink[p[0]][p[1]] = true;
      for (int y = 0; y < side - 1; ++y)
        for (int x = 0; x < side - 1; ++x)
          if (ink[y][x])
            std::fill_n(&src[((y+1)*side+x+1)*3],3,20);
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x)
          if (ink[y][x]) std::fill_n(&src[(y*side+x)*3],3,220);
      dl_rect r{0,0,side,side};
      for (int rotation = 0; rotation < 4; ++rotation)
        for (int mutation = 0; mutation < 3; ++mutation) {
          auto probe = src;
          // These neighboring shadow pixels can otherwise trigger a later
          // three-row notch over the confirmed gap in two orientations.
          if (mutation == 1) std::fill_n(&probe[(5*side+6)*3],3,20);
          if (mutation == 2) std::fill_n(&probe[(6*side+5)*3],3,20);
          for (int turn = 0; turn < rotation; ++turn) {
            std::vector<uint8_t> rotated(probe.size());
            for (int y = 0; y < side; ++y)
              for (int x = 0; x < side; ++x)
                for (int c = 0; c < 3; ++c)
                  rotated[((side-1-x)*side+y)*3+c] = probe[(y*side+x)*3+c];
            probe.swap(rotated);
          }
          CHECK(dl_scale3(context.get(),probe.data(),side,side,side*3,out.data(),
                          side*9,&r,1,nullptr)==0);
          for (int y = 12; y < 15; ++y)
            for (int x = 12; x < 15; ++x)
              for (int c = 0; c < 3; ++c)
                CHECK(out[(y*side*3+x)*3+c]==64);
        }
    }
    // Orthogonal two-pixel stems and one-pixel bars keep square joins/caps.
    // Minimal valid/clipped rasters exercise the extra cap-witness bounds.
    if (context->impl.model.version >= 4) {
      constexpr int cells[8][2]={{0,0},{1,0},{-1,-1},{-1,1},
                                  {2,-1},{2,1},{-1,0},{2,0}};
      for (int width : {5,6,13}) {
        int height=width==13?13:5, cx=width==13?5:2, cy=width==13?6:2;
        for (bool inverse : {false,true}) {
          uint8_t fg=inverse?32:220, bg=inverse?220:32;
          std::vector<uint8_t> src(width*height*3,bg),out(width*height*27);
          for (int y=std::max(0,cy-3);y<std::min(height,cy+4);++y)
            for (int x=cx;x<cx+2;++x) std::fill_n(&src[(y*width+x)*3],3,fg);
          for (int x=cx-1;x<=cx+2;++x) std::fill_n(&src[(cy*width+x)*3],3,fg);
          dl_rect r{0,0,width,height};
          CHECK(dl_scale3(context.get(),src.data(),width,height,width*3,
                          out.data(),width*9,&r,1,nullptr)==0);
          dl_detail_stats detail{};detail.struct_size=sizeof(detail);
          CHECK(dl_last_details(context.get(),&detail)==0);
          if (width==5) {
            CHECK(detail.ufo_joins==0);
            continue;
          }
          CHECK(detail.ufo_joins>=8);
          for (auto &p : cells) {
            int x=cx+p[0],y=cy+p[1];
            for (int oy=0;oy<3;++oy)
              for (int ox=0;ox<3;++ox)
                for (int k=0;k<3;++k)
                  CHECK(out[((y*3+oy)*width*3+x*3+ox)*3+k]==src[(y*width+x)*3+k]);
          }
        }
      }
    }
    // Masked mixed-width joins tolerate row shading and irrelevant far caps.
    // Minimum 5x5 frames exercise every outer-witness read under sanitizers.
    if (context->impl.model.version >= 4) {
      for (int side : {5,13})
        for (int layout : {0,1,2})
          for (int rotation=0; rotation<4; ++rotation) {
            int c=side/2, stem_side=layout==2?-1:1;
            std::vector<uint8_t> src(side*side*3,72),out(side*side*27);
            auto put=[&](int u,int v,int value) {
              int x=c+u,y=c-v;
              for (int r=0;r<rotation;++r) {int t=x;x=side-1-y;y=t;}
              std::fill_n(&src[(y*side+x)*3],3,value);
            };
            for (int u=-2;u<=2;++u) put(u,0,156+(layout==0?12*u:0));
            if (layout==0) {
              for (int u=-1;u<=1;++u) put(u,1,156+12*u);
              for (int v : {-1,-2}) put(0,v,156);
              put(0,2,24);
            } else {
              for (int v : {-1,-2})
                for (int u : {0,stem_side}) put(u,v,156-12*v);
            }
            int left=layout==0?-1:(stem_side==1?-1:-2),
                right=layout==0?1:(stem_side==1?2:1);
            for (int u : {left,right})
              for (int v : {-1,-2}) put(u,v,24);
            if (layout!=0) put(left,-2,72);
            dl_rect hint{0,0,side,side};
            CHECK(dl_scale3(context.get(),src.data(),side,side,side*3,
                            out.data(),side*9,&hint,1,nullptr)==0);
            for (int y=c*3;y<c*3+3;++y)
              for (int x=c*3;x<c*3+3;++x)
                for (int k=0;k<3;++k) CHECK(out[(y*side*3+x)*3+k]==156);
          }
    }
    // A thin same-color junction may be a gap between glyphs above a
    // textured field. Only its central block is preserved. Minimal 5x5
    // inputs exercise all radius-two witnesses under sanitizers.
    if (context->impl.model.version >= 4) {
      constexpr uint8_t colors[8][3]={{0,16,32},{190,137,60},{16,48,64},
          {24,48,72},{24,56,80},{8,40,48},{16,40,64},{24,64,80}};
      constexpr int ids[5][5]={{0,1,0,1,0},{0,1,0,1,0},{0,0,0,0,0},
                              {2,2,3,4,4},{5,6,3,4,7}};
      for (int side : {5,13})
        for (bool inverse : {false,true})
          for (int rotation=0;rotation<4;++rotation) {
            const int c=side/2;
            std::vector<uint8_t> src(side*side*3),out(side*side*27);
            for (int y=0;y<side;++y)
              for (int x=0;x<side;++x)
                for (int k=0;k<3;++k)
                  src[(y*side+x)*3+k]=inverse?255-colors[0][k]:colors[0][k];
            for (int v=-2;v<=2;++v)
              for (int u=-2;u<=2;++u) {
                int x=c+u,y=c+v;
                for (int r=0;r<rotation;++r) {int t=x;x=side-1-y;y=t;}
                for (int k=0;k<3;++k) {
                  auto value=colors[ids[v+2][u+2]][k];
                  src[(y*side+x)*3+k]=inverse?255-value:value;
                }
              }
            dl_rect hint{0,0,side,side};
            CHECK(dl_scale3(context.get(),src.data(),side,side,side*3,
                            out.data(),side*9,&hint,1,nullptr)==0);
            for (int y=c*3;y<c*3+3;++y)
              for (int x=c*3;x<c*3+3;++x)
                for (int k=0;k<3;++k)
                  CHECK(out[(y*side*3+x)*3+k]==src[(c*side+c)*3+k]);
          }
    }
    // A graded K/R notch retains the established wedge coverage but uses
    // each source row's foreground/shadow. Minimum5x5 covers outer witnesses.
    if (context->impl.model.version >= 4) {
      for (int side : {5,17})
        for (int alpha : {0,64,128,192})
          for (bool inverse : {false,true}) {
            int c=side/2;
            auto make=[&](bool graded) {
              std::vector<uint8_t> v(side*side*3,inverse?255:0);
              auto put=[&](int x,int y,int value) {
                std::fill_n(&v[(y*side+x)*3],3,inverse?255-value:value);
              };
              const int base=graded?128:252;
              put(c-2,c,base);put(c-1,c,base);put(c,c,(base*alpha+128)/256);
              put(c,c-1,base-(graded?16:0));put(c,c+1,base+(graded?16:0));
              put(c+1,c-2,base-(graded?32:0));put(c+1,c+2,base+(graded?32:0));
              return v;
            };
            auto flat=make(false),graded=make(true);
            std::vector<uint8_t> ref(side*side*27),out(ref.size());
            dl_rect hint{0,0,side,side};
            CHECK(dl_scale3(context.get(),flat.data(),side,side,side*3,
                            ref.data(),side*9,&hint,1,nullptr)==0);
            CHECK(dl_scale3(context.get(),graded.data(),side,side,side*3,
                            out.data(),side*9,&hint,1,nullptr)==0);
            for (int row=-1;row<=1;++row)
              for (int oy=0;oy<3;++oy)
                for (int ox=0;ox<3;++ox) {
                  int pos=((c+row)*3+oy)*side*9+c*9+ox*3;
                  int cov=inverse?255-ref[pos]:ref[pos];
                  int wanted=(cov*(128+row*16)+126)/252;
                  if(inverse)wanted=255-wanted;
                  CHECK(std::abs(int(out[pos])-wanted)<=2);
                }
          }
    }
    // Extreme text mixtures keep their average as a small geometric cut,
    // including inverse polarity and all corner orientations.
    if (context->impl.model.version >= 4) {
      constexpr int side=9;
      for (int level : {16,232,240,247})
        for (bool inverse : {false,true})
          for (int rotation=0;rotation<4;++rotation) {
            std::vector<uint8_t> src(side*side*3,inverse?255:0),out(side*side*27);
            for (int y=0;y<5;++y)
              for (int x=0;x<5;++x)
                std::fill_n(&src[(y*side+x)*3],3,inverse?0:255);
            const int expected=inverse?255-level:level;
            std::fill_n(&src[(4*side+4)*3],3,uint8_t(expected));
            for (int turn=0;turn<rotation;++turn) {
              auto copy=src;
              for (int y=0;y<side;++y)
                for (int x=0;x<side;++x)
                  for (int c=0;c<3;++c)
                    src[((side-1-x)*side+y)*3+c]=copy[(y*side+x)*3+c];
            }
            dl_rect r{0,0,side,side};
            CHECK(dl_scale3(context.get(),src.data(),side,side,side*3,out.data(),
                            side*9,&r,1,nullptr)==0);
            int sum=0,lo=255,hi=0;
            for (int y=12;y<15;++y)
              for (int x=12;x<15;++x) {
                int v=out[(y*side*3+x)*3];sum+=v;lo=std::min(lo,v);hi=std::max(hi,v);
              }
            CHECK(std::abs(sum-expected*9)<=9);
            CHECK(lo<hi);
            dl_detail_stats detail{};detail.struct_size=sizeof(detail);
            CHECK(dl_last_details(context.get(),&detail)==0);
            CHECK(detail.ufo_corners>=1);
          }
    }
    // Explicit checker de-dithering is disabled for all palette policies.
    {
      const int width = 40, height = 40;
      std::vector<uint8_t> data(width * height * 3);
      auto setup = [&](int a, int b) {
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x) {
            auto p = &data[(y * width + x) * 3];
            p[0] = p[1] = 0;
            p[2] = uint8_t((x + y) % 2 ? a : b);
          }
      };
      auto bank = [&](bool neighbors = false) {
        lut::Frame f;
        f.prepare(data.data(), width, height, width * 3, nullptr, 0, neighbors);
        uint32_t c[9];
        f.core(20, 20, c);
        return f.bank(20, 20, c).index;
      };
      setup(85, 170);
      CHECK(bank() == 0);
      data[2] = 128;
      CHECK(bank() == 0); // An intervening observed blue shade.
      setup(0, 255);
      CHECK(bank() == 0);
      setup(85, 170);
      for (int i = 0; i < 17; ++i) {
        data[i * 3] = uint8_t(i);
        data[i * 3 + 1] = uint8_t(i + 1);
        data[i * 3 + 2] = uint8_t(i + 2);
      }
      CHECK(bank() == 0);
      // Cross-hue nearest neighbors: channel deltas have opposite signs.
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
          auto p = &data[(y * width + x) * 3];
          p[0] = 40;
          p[1] = uint8_t((x + y) % 2 ? 100 : 160);
          p[2] = uint8_t((x + y) % 2 ? 160 : 100);
        }
      CHECK(bank() == 0); // Historical versions retain same-hue policy.
      CHECK(bank(true) == 0);
      data[0] = 40;
      data[1] = data[2] = 130;
      CHECK(bank(true) == 0); // Intermediate palette color blocks this edge.
      setup(0, 255);
      CHECK(bank(true) == 0);
    }
    std::atomic<bool> good{true};
    auto worker = [&] {
      auto c = dl_create(argv[1]);
      if (!c) {
        good = false;
        return;
      }
      uint8_t s[3] = {12, 84, 168}, out[27];
      for (int n = 0; n < 5; ++n) {
        if (dl_scale3(c, s, 1, 1, 3, out, 9, nullptr, 0, nullptr))
          good = false;
        for (int i = 0; i < 9; ++i)
          for (int k = 0; k < 3; ++k)
            if (out[i * 3 + k] != s[k])
              good = false;
      }
      dl_destroy(c);
    };
    std::thread a(worker), b(worker);
    a.join();
    b.join();
    CHECK(good);
    std::cout
        << "HQ fallback, tiny rasters, strides, source preservation, parameter "
           "rejection, neutral colors, context reuse and concurrency passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
