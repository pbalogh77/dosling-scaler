#include "model.hpp"
#if defined(DL_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace { constexpr uint64_t plane_core_masks[] = {0xb0000d08ac8ULL, 0x30100000000ULL, 0x510000ULL, 0x1010100110001ULL}; }
namespace {
// Direct source-color borrowing requires two coherent color groups across
// the actual5x5 footprint. A binary key alone can merge unrelated surfaces.
bool contour_source(const lut::Frame &f,int x,int y,const uint32_t *core,
                    uint8_t key,uint16_t outer) {
  if(x<2 || y<2 || x>=f.w-2 || y>=f.h-2) return false;
  const unsigned threshold=f.threshold[size_t(y)*f.w+x];
  const int same=std::min(576u,std::max(144u,threshold/4));
  const int separated=std::max(3600u,threshold*2);
#if defined(DL_NEON) && defined(__aarch64__)
  const auto center4=vdupq_n_u32(core[4]),mask255=vdupq_n_u32(255);
  const uint32x4_t bits={1,2,4,8};
  auto lo=vdupq_n_u8(255),hi=vdupq_n_u8(0);
  auto accept4=[&](uint32x4_t values,unsigned pattern) {
    auto r=vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(values,16)),vreinterpretq_s32_u32(vshrq_n_u32(center4,16)));
    auto g=vsubq_s32(vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(values,8),mask255)),vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(center4,8),mask255)));
    auto b=vsubq_s32(vreinterpretq_s32_u32(vandq_u32(values,mask255)),vreinterpretq_s32_u32(vandq_u32(center4,mask255)));
    auto d=vreinterpretq_u32_s32(vmlaq_s32(vmlaq_s32(vmulq_s32(r,r),g,g),b,b));
    auto other=vcgtq_u32(vandq_u32(vdupq_n_u32(pattern),bits),vdupq_n_u32(0));
    auto valid=vbslq_u32(other,vcgeq_u32(d,vdupq_n_u32(separated)),vcleq_u32(d,vdupq_n_u32(same)));
    if(vminvq_u32(valid)!=UINT32_MAX)return false;
    {
      auto selected=vreinterpretq_u8_u32(other),rgb=vreinterpretq_u8_u32(values);
      lo=vminq_u8(lo,vbslq_u8(selected,rgb,vdupq_n_u8(255)));
      hi=vmaxq_u8(hi,vbslq_u8(selected,rgb,vdupq_n_u8(0)));
    }
    return true;
  };
  if(!accept4(vld1q_u32(core),key) || !accept4(vld1q_u32(core+5),key>>4))return false;
  const size_t center=size_t(y+2)*f.pitch+x+2;
  const uint32x4_t ring[4]={vld1q_u32(&f.pixels[center-2*f.pitch-2]),
    {f.pixels[center-2*f.pitch+2],f.pixels[center-f.pitch-2],f.pixels[center-f.pitch+2],f.pixels[center-2]},
    {f.pixels[center+2],f.pixels[center+f.pitch-2],f.pixels[center+f.pitch+2],f.pixels[center+2*f.pitch-2]},
    vld1q_u32(&f.pixels[center+2*f.pitch-1])};
  for(int i=0;i<4;++i)if(!accept4(ring[i],outer>>(i*4)))return false;
  {
    auto low=vmin_u8(vget_low_u8(lo),vget_high_u8(lo));
    auto high=vmax_u8(vget_low_u8(hi),vget_high_u8(hi));
    low=vmin_u8(low,vreinterpret_u8_u32(vdup_n_u32(vget_lane_u32(vreinterpret_u32_u8(low),1))));
    high=vmax_u8(high,vreinterpret_u8_u32(vdup_n_u32(vget_lane_u32(vreinterpret_u32_u8(high),1))));
    auto a=lut::rgb(vget_lane_u32(vreinterpret_u32_u8(low),0));
    auto b=lut::rgb(vget_lane_u32(vreinterpret_u32_u8(high),0));
    const int r=(b.r-a.r+1)/2,g=(b.g-a.g+1)/2,bb=(b.b-a.b+1)/2;
    if(r*r+g*g+bb*bb>same)return false;
  }
#else
  int lo[3]={255,255,255},hi[3]={0,0,0};
  auto accept=[&](uint32_t value,bool other) {
    int d=lut::distance(value,core[4]);
    if(other?d<separated:d>same) return false;
    if(other) {
      auto c=lut::rgb(value);int v[]={c.r,c.g,c.b};
      for(int i=0;i<3;++i) {lo[i]=std::min(lo[i],v[i]);hi[i]=std::max(hi[i],v[i]);}
    }
    return true;
  };
  int bit=0;
  for(int i=0;i<9;++i)if(i!=4)
    if(!accept(core[i],(key>>bit++)&1u)) return false;
  bit=0;const size_t center=size_t(y+2)*f.pitch+x+2;
  for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx)
    if(std::abs(dx)==2 || std::abs(dy)==2)
      if(!accept(f.pixels[center+dy*f.pitch+dx],(outer>>bit++)&1u)) return false;
  {
    int radius=0;
    for(int i=0;i<3;++i) {const int d=(hi[i]-lo[i]+1)/2;radius+=d*d;}
    if(radius>same) return false;
  }
#endif
  return true;
}
}

#include <new>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
namespace {
bool buffer(int w, int h, size_t stride, size_t &size, size_t channels) {
  if (w <= 0 || h <= 0 || size_t(w) > SIZE_MAX / channels ||
      stride < size_t(w) * channels ||
      size_t(h - 1) > (SIZE_MAX - size_t(w) * channels) / stride)
    return false;
  size = size_t(h - 1) * stride + size_t(w) * channels;
  return true;
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return an > UINTPTR_MAX - x || bn > UINTPTR_MAX - y ||
         (x < y + bn && y < x + an);
}
} // namespace
extern "C" {
dl_context *dl_create(const char *path) {
  if (!path)
    return nullptr;
  try {
    auto m = lut::load(path);
    if (m.version >= 4)
      lut::warm_stencils();
    auto c = new dl_context;
    c->impl.model = std::move(m);
    c->impl.bytes = lut::model_bytes(c->impl.model);
    return c;
  } catch (...) {
    return nullptr;
  }
}
dl_context *dl_create_from_memory(const void *bytes, size_t n) {
  try {
    auto m = lut::load_memory(bytes, n);
    if (m.version >= 4)
      lut::warm_stencils();
    auto c = new dl_context;
    c->impl.model = std::move(m);
    c->impl.bytes = n;
    return c;
  } catch (...) {
    return nullptr;
  }
}
void dl_destroy(dl_context *c) { delete c; }
int dl_last_details(const dl_context *c, dl_detail_stats *d) {
  if (!c || !d || d->struct_size != sizeof(dl_detail_stats))
    return 1;
  *d = c->impl.details;
  return 0;
}
size_t dl_model_bytes(const dl_context *c) { return c ? c->impl.bytes : 0; }
const char *dl_backend(void) {
#if defined(DL_NEON) && (defined(__aarch64__) || defined(__ARM_NEON))
  return "neon";
#else
  return "scalar";
#endif
}
static int scale_impl(dl_context *c, const uint8_t *src, int w, int h,
                      size_t ss, uint8_t *dst, size_t ds,
                      const dl_rect *regions, size_t nr, dl_stats *stats,
                      bool bgra) {
  size_t sn = 0, dn = 0;
  if (!c || !src || !dst || w > INT32_MAX / 3 || h > INT32_MAX / 3 ||
      !buffer(w, h, ss, sn, bgra ? 4 : 3) ||
      !buffer(w * 3, h * 3, ds, dn, bgra ? 4 : 3) ||
      overlap(src, sn, dst, dn) || (nr && !regions))
    return 1;
  for (size_t i = 0; i < nr; ++i) {
    auto r = regions[i];
    if (r.x0 < 0 || r.y0 < 0 || r.x0 >= r.x1 || r.y0 >= r.y1 || r.x1 > w ||
        r.y1 > h)
      return 1;
  }
  try {
    auto &f = c->impl.frame;
    auto &m = c->impl.model;
    f.prepare(src, w, h, ss, regions, nr, m.version >= 3, bgra);
    if (m.version >= 4)
      lut::prepare_ufo(f);
    struct alignas(64) Partial {
      dl_stats s{};
      dl_detail_stats d{};
    };
    std::array<Partial, 8> partials{};
    auto render = [&](int y0, int y1, Partial &partial) {
      auto &s = partial.s;
      auto &detail = partial.d;
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
          uint32_t core[9], out[9];
          f.core(x, y, core);
          const int special =
              m.version >= 4 ? lut::ufo_block(f, x, y, core, out) : 0;
          if (special) {
            detail.ufo_corners += special == 1;
            detail.ufo_joins += special == 2;
            detail.ufo_wedges += special == 3;
          }
          uint32_t variation = 0;
          for (int i = 0; i < 9; ++i)
            variation |= core[i] ^ core[4];
          if (!special && !variation) {
            uint32_t v = core[4];
            if (bgra) {
              uint32_t row[3] = {v | 0xff000000u, v | 0xff000000u,
                                 v | 0xff000000u};
              for (int oy = 0; oy < 3; ++oy)
                std::memcpy(dst + size_t(y * 3 + oy) * ds + size_t(x) * 12, row,
                            12);
            } else {
              uint8_t row[9];
              for (int i = 0; i < 3; ++i) {
                row[i * 3] = uint8_t(v >> 16);
                row[i * 3 + 1] = uint8_t(v >> 8);
                row[i * 3 + 2] = uint8_t(v);
              }
              for (int oy = 0; oy < 3; ++oy)
                std::memcpy(dst + size_t(y * 3 + oy) * ds + size_t(x) * 9, row,
                            9);
            }
            ++s.pixels;
            s.text += f.text[size_t(y) * w + x] != 0;
            continue;
          }
          // Only ordinary binary plane/diagonal exceptions may use bank2.
          // Explicit checker de-dithering is disabled, independent of palette.
          auto key = f.keys[size_t(y) * w + x];
          auto selected = special ? lut::Frame::BankSelection{1} : f.bank(x, y, core,
              m.version >= 4 &&
              ((plane_core_masks[key>>6]>>(key&63))&1u) &&
              m.primary[512 + key].extended != lut::NO_EXT);
          int bank = selected.index;
          bool hq = false;
          if (!special) {
            auto entry = m.primary[bank * 256 + f.keys[size_t(y) * w + x]];
            uint16_t recipe = entry.recipe;
            if (entry.extended != lut::NO_EXT) {
              uint16_t extended;
              if (m.version >= 4) {
                const uint16_t outer=selected.outer_ready ? selected.outer :
                    f.outer5(x,y,f.threshold[size_t(y)*w+x]);
                extended=m.exceptions[entry.extended].lookup(outer);
                if(extended!=lut::NO_EXT && bank!=2 && (m.flags[extended]&8u) &&
                   !contour_source(f,x,y,core,key,outer)) extended=lut::NO_EXT;
                detail.exception_hits += extended != lut::NO_EXT;
                detail.exception_misses += extended == lut::NO_EXT;
              } else
                extended =
                    m.extended[entry.extended & 0x7fff]
                              [f.outer(x, y, f.threshold[size_t(y) * w + x],
                                       entry.extended >> 15)];
              if (extended != lut::NO_EXT)
                recipe = extended;
              ++s.extended;
            }
            if (m.flags[recipe] & 5) {
              // A repeated ordinary color instruction has the same result at
              // all nine positions. Evaluate once and skip unused YUV loads.
              auto value = (m.flags[recipe] & 1)
                               ? core[4]
                               : lut::color_op(m.ops[m.blocks[recipe].op[0]], core);
              std::fill_n(out, 9, value);
            } else {
              uint32_t yuv[9];
              size_t center = size_t(y + 2) * f.pitch + x + 2;
              for (int i = 0; i < 9; ++i)
                yuv[i] = f.yuv[center + lut::cy[i] * f.pitch + lut::cx[i]];
              if (m.flags[recipe] & 2) {
                ds_hq3x_block_cached(core, yuv,
                                     f.fallback_keys[size_t(y) * w + x], out);
                hq = true;
              } else
                lut::evaluate(m, m.blocks[recipe], core, out, &hq, yuv,
                              f.fallback_keys[size_t(y) * w + x]);
            }
          } // Learned lookup is skipped when a hardcoded geometry rule claims the
            // block.
          for (int oy = 0; oy < 3; ++oy) {
            auto row =
                dst + size_t(y * 3 + oy) * ds + size_t(x) * (bgra ? 12 : 9);
            if (bgra) {
              uint32_t packed[3] = {out[oy * 3] | 0xff000000u,
                                    out[oy * 3 + 1] | 0xff000000u,
                                    out[oy * 3 + 2] | 0xff000000u};
              std::memcpy(row, packed, 12);
              continue;
            }
            for (int ox = 0; ox < 3; ++ox) {
              uint32_t v = out[oy * 3 + ox];
              row[ox * 3] = uint8_t(v >> 16);
              row[ox * 3 + 1] = uint8_t(v >> 8);
              row[ox * 3 + 2] = uint8_t(v);
            }
          }
          ++s.pixels;
          s.hq_fallback += hq;
          s.text += f.text[size_t(y) * w + x] != 0;
          s.shade_checker += bank == 2;
        }
    };
    unsigned jobs = 1;
#ifdef __APPLE__
    jobs = c->impl.workers ? c->impl.workers
                           : (size_t(w) * h >= 200000  ? 4
                              : size_t(w) * h >= 32000 ? 2
                                                       : 1);
    jobs = std::min(jobs, unsigned(h));
    if (jobs > 1) {
      struct Work {
        decltype(render) *fn;
        Partial *parts;
        int height;
        unsigned jobs;
      } work{&render, partials.data(), h, jobs};
      dispatch_apply_f(jobs,
                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       &work, [](void *p, size_t i) {
                         auto &w = *static_cast<Work *>(p);
                         (*w.fn)(int(i * w.height / w.jobs),
                                 int((i + 1) * w.height / w.jobs), w.parts[i]);
                       });
    } else
#endif
      render(0, h, partials[0]);
    dl_stats s{};
    dl_detail_stats detail{};
    detail.struct_size = sizeof(detail);
    for (unsigned i = 0; i < jobs; ++i) {
      auto a = partials[i].s;
      auto d = partials[i].d;
      s.pixels += a.pixels;
      s.extended += a.extended;
      s.hq_fallback += a.hq_fallback;
      s.text += a.text;
      s.shade_checker += a.shade_checker;
      detail.ufo_corners += d.ufo_corners;
      detail.ufo_joins += d.ufo_joins;
      detail.ufo_wedges += d.ufo_wedges;
      detail.exception_hits += d.exception_hits;
      detail.exception_misses += d.exception_misses;
    }
    c->impl.details = detail;
    if (stats)
      *stats = s;
    return 0;
  } catch (const std::bad_alloc &) {
    return 2;
  } catch (...) {
    return 1;
  }
}
int dl_set_workers(dl_context *c, unsigned n) {
  if (!c || n > 8)
    return 1;
  c->impl.workers = n;
  return 0;
}
int dl_scale3(dl_context *c, const uint8_t *s, int w, int h, size_t ss,
              uint8_t *d, size_t ds, const dl_rect *r, size_t n,
              dl_stats *stats) {
  return scale_impl(c, s, w, h, ss, d, ds, r, n, stats, false);
}
int dl_scale3_bgra(dl_context *c, const uint8_t *s, int w, int h, size_t ss,
                   uint8_t *d, size_t ds, const dl_rect *r, size_t n,
                   dl_stats *stats) {
  return scale_impl(c, s, w, h, ss, d, ds, r, n, stats, true);
}
}
