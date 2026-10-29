// Dosling scaler: source-supported AA geometry plus optional glyph stencils.
// General AA uses actual two-surface/coverage evidence, never OCR or coordinates.
#include "model.hpp"
#include <numeric>
#if defined(DL_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace lut {
namespace {
constexpr uint8_t FIELD_GAP_PRIORITY = 0x80;
constexpr int AA_ENDPOINT_CAP = 1296;
// Research builds can compare support/monotonicity footprints. The released
// defaults keep the full RGB evidence while ordering only the local core.
#ifndef DL_GENERAL_AA_SUPPORT
#define DL_GENERAL_AA_SUPPORT 5
#endif
#ifndef DL_GENERAL_MONOTONE_RADIUS
#define DL_GENERAL_MONOTONE_RADIUS 1
#endif
static_assert(DL_GENERAL_AA_SUPPORT==3 || DL_GENERAL_AA_SUPPORT==5);
static_assert(DL_GENERAL_MONOTONE_RADIUS>=0 && DL_GENERAL_MONOTONE_RADIUS<=2);
static_assert(DL_GENERAL_AA_SUPPORT/2>=DL_GENERAL_MONOTONE_RADIUS);
uint32_t pixel(const Frame &f, int x, int y) {
  return f.pixels[size_t(y + 2) * f.pitch + x + 2];
}
Color mean(Color a, Color b) {
  return {(a.r + b.r + 1) / 2, (a.g + b.g + 1) / 2, (a.b + b.b + 1) / 2};
}
uint32_t blend(Color a, Color b, int q) {
  return packed({clip(a.r + floor_div((b.r - a.r) * q + 128, 256)),
                 clip(a.g + floor_div((b.g - a.g) * q + 128, 256)),
                 clip(a.b + floor_div((b.b - a.b) * q + 128, 256))});
}
// Wider evidence can choose geometry, but cannot extend the RGB range of
// the actual source3x3 core. Apply only to reconstructed blocks; own-source
// copies and the convex local lookup/HQ paths are bounded by construction.
void bound_source_range(const uint32_t *core,uint32_t *out) {
#if defined(DL_NEON) && defined(__aarch64__)
  const auto a=vreinterpretq_u8_u32(vld1q_u32(core)),b=vreinterpretq_u8_u32(vld1q_u32(core+4));
  const auto mn=vminq_u8(a,b),mx=vmaxq_u8(a,b);
  auto low=vmin_u8(vget_low_u8(mn),vget_high_u8(mn));
  auto high=vmax_u8(vget_low_u8(mx),vget_high_u8(mx));
  low=vmin_u8(low,vext_u8(low,low,4));high=vmax_u8(high,vext_u8(high,high,4));
  const auto last=vreinterpret_u8_u32(vdup_n_u32(core[8]));
  low=vmin_u8(low,last);high=vmax_u8(high,last);
  const auto low4=vcombine_u8(low,low),high4=vcombine_u8(high,high);
  for(int i:{0,4})vst1q_u32(out+i,vreinterpretq_u32_u8(vminq_u8(high4,vmaxq_u8(low4,vreinterpretq_u8_u32(vld1q_u32(out+i))))));
  const auto value=vreinterpret_u8_u32(vdup_n_u32(out[8]));
  out[8]=vget_lane_u32(vreinterpret_u32_u8(vmin_u8(high,vmax_u8(low,value))),0);
#else
  Color low=rgb(core[0]),high=low;
  for(int i=1;i<9;++i){const auto c=rgb(core[i]);low={std::min(low.r,c.r),std::min(low.g,c.g),std::min(low.b,c.b)};high={std::max(high.r,c.r),std::max(high.g,c.g),std::max(high.b,c.b)};}
  for(int i=0;i<9;++i){const auto c=rgb(out[i]);out[i]=packed({clip(c.r,low.r,high.r),clip(c.g,low.g,high.g),clip(c.b,low.b,high.b)});}
#endif
}
bool near(uint32_t a, uint32_t b) { return distance(a, b) <= 144; }
// Componentwise median of six continued surface witnesses. The fixed
// network avoids sorting-library branches in the rare shaded-endpoint path.
Color continued_endpoint(const Frame &f,int x,int y,int sx,int sy) {
  const uint32_t samples[6]={pixel(f,x+2*sx,y),pixel(f,x,y+2*sy),
    pixel(f,x+sx,y+sy),pixel(f,x+2*sx,y+sy),
    pixel(f,x+sx,y+2*sy),pixel(f,x+2*sx,y+2*sy)};
  constexpr int network[][2]={{1,2},{4,5},{0,2},{3,5},{0,1},{3,4},
                             {2,5},{0,3},{1,4},{2,4},{1,3},{2,3}};
#if defined(DL_NEON) && defined(__aarch64__)
  uint8x8_t values[6];
  for(int i=0;i<6;++i)values[i]=vreinterpret_u8_u32(vdup_n_u32(samples[i]));
  for(auto &pair:network) {
    auto a=values[pair[0]],b=values[pair[1]];
    values[pair[0]]=vmin_u8(a,b);values[pair[1]]=vmax_u8(a,b);
  }
  return rgb(vget_lane_u32(vreinterpret_u32_u8(vrhadd_u8(values[2],values[3])),0));
#else
  Color values[6];for(int i=0;i<6;++i)values[i]=rgb(samples[i]);
  for(auto &pair:network) {
    auto a=values[pair[0]],b=values[pair[1]];
    values[pair[0]]={std::min(a.r,b.r),std::min(a.g,b.g),std::min(a.b,b.b)};
    values[pair[1]]={std::max(a.r,b.r),std::max(a.g,b.g),std::max(a.b,b.b)};
  }
  return mean(values[2],values[3]);
#endif
}
// A straight partially covered row/column between two repeated surfaces.
// Codes 9/10 fit the existing byte stencil map; no new persistent buffer.
// All 25 samples must fit two repeated endpoint surfaces and a common RGB
// mixture line. This path is text-hint scoped; it does not smooth checkerboards.
Color median3(Color a,Color b,Color c) {
  auto m=[](int x,int y,int z){return x+y+z-std::min({x,y,z})-std::max({x,y,z});};
  return {m(a.r,b.r,c.r),m(a.g,b.g,c.g),m(a.b,b.b,c.b)};
}
// Two adjacent, collinear partial-coverage witnesses inside a purported
// opaque field can be the continuation of another edge, rather than noise.
bool repeated_intermediate(uint32_t field) {
  // A 5x5 field fits in 25 bits. Mask row starts to exclude horizontal
  // wraparound; the other shift tests vertical adjacency. No extra pass.
  constexpr uint32_t row_starts=0x108421u;
  return (field&(field<<1)&~row_starts) || (field&(field<<5));
}
constexpr int AA_PHASE_TOLERANCE=16,AA_ENDPOINT_TOLERANCE=16;
constexpr int AA_FIELD_CAP=1296,AA_FIELD_DIVISOR=64;
struct Axial {uint32_t fg,bg;int alpha;};
bool axial(const Frame &f,int x,int y,bool horizontal,Axial &out) {
  if(x<2 || y<2 || x>=f.w-2 || y>=f.h-2)return false;
  const int nx=horizontal?1:0,ny=horizontal?0:1,tx=ny,ty=nx;
  auto at=[&](int u,int v){return pixel(f,x+u*tx+v*nx,y+u*ty+v*ny);};
  auto estimate=[&](int v){return median3(rgb(at(-1,v)),rgb(at(0,v)),rgb(at(1,v)));};
  auto fg=estimate(-1),bg=estimate(1);const auto fp=packed(fg),bp=packed(bg);
  const int dr=fg.r-bg.r,dg=fg.g-bg.g,db=fg.b-bg.b,n=dr*dr+dg*dg+db*db;
  if(n<6912)return false;
  // A source channel denotes an 8-bit rounding interval. Project its half-
  // level uncertainty onto the common RGB line before classifying a faint
  // continued mixture. This matters near the minimum contrast threshold.
  const int continuation_margin=128*(std::abs(dr)+std::abs(dg)+std::abs(db));
  const int same=std::min(AA_FIELD_CAP,std::max(144,n/AA_FIELD_DIVISOR));
  auto ratio=[&](uint32_t value){auto c=rgb(value);return floor_div(((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2,n);};
  const int alpha=ratio(at(0,0));if(alpha<16 || alpha>240)return false;
  const int residual=std::min(1024,std::max(144,n/64));int fa=0,ba=0;uint32_t mid_fg=0,mid_bg=0;
  for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u) {
    auto c=at(u,v);
    if(v) {
      if(distance(c,v<0?fp:bp)>same)return false;
      const auto cc=rgb(c);const int projection=((cc.r-bg.r)*dr+(cc.g-bg.g)*dg+(cc.b-bg.b)*db)*256+n/2;
      if(projection>=16*n+n/2-continuation_margin && projection<=240*n+n/2+continuation_margin) {
        const int q=ratio(c);if(distance(c,blend(bg,fg,q))<=144)(v<0?mid_fg:mid_bg)|=1u<<((v+2)*5+u+2);
      }
      fa+=v<0 && projection>=(256-AA_ENDPOINT_TOLERANCE)*n && projection<(257+AA_ENDPOINT_TOLERANCE)*n;
      ba+=v>0 && projection>=-AA_ENDPOINT_TOLERANCE*n && projection<(1+AA_ENDPOINT_TOLERANCE)*n;
    }
    else {const int a=ratio(c);if(std::abs(a-alpha)>AA_PHASE_TOLERANCE || distance(c,blend(bg,fg,a))>residual)return false;}
  }
  if(fa<8 || ba<8 || ((repeated_intermediate(mid_fg) || repeated_intermediate(mid_bg))))return false;
  out={fp,bp,alpha};return true;
}
struct Joint {uint32_t fg,bg;int a,b;};
bool joint(const Frame &f,int x,int y,int orientation,Joint &out,bool validate=true) {
  if(x<2 || y<2 || x>=f.w-2 || y>=f.h-2)return false;
  int dx=orientation&1?-1:1,dy=orientation&2?-1:1;
  auto at=[&](int u,int v){return pixel(f,x+u*dx,y+v*dy);};
  const auto f0=at(-1,-1),f1=at(1,-1),f2=at(-1,1),b0=at(1,1);
  if(validate && (!near(f0,f1) || !near(f0,f2) || distance(f0,b0)<1800))return false;
  const auto fg=median3(rgb(f0),rgb(f1),rgb(f2));
  const auto bg=median3(rgb(b0),rgb(at(2,1)),rgb(at(1,2)));
  const int dr=fg.r-bg.r,dg=fg.g-bg.g,db=fg.b-bg.b,n=dr*dr+dg*dg+db*db;
  if(n<6912)return false;
  // A source channel denotes an 8-bit rounding interval. Project its half-
  // level uncertainty onto the common RGB line before classifying a faint
  // continued mixture. This matters near the minimum contrast threshold.
  const int continuation_margin=128*(std::abs(dr)+std::abs(dg)+std::abs(db));
  auto ratio=[&](uint32_t value){auto c=rgb(value);return floor_div(((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2,n);};
  const int a=ratio(packed(mean(rgb(at(1,0)),rgb(at(2,0))))),
            b=ratio(packed(mean(rgb(at(0,1)),rgb(at(0,2)))));
  if(a<0 || a>256 || b<0 || b>256)return false;
  const int expected=256-((256-a)*(256-b)+128)/256;
  // Two partial arms can prove a very faint product corner even when
  // its own RGB average rounds to an endpoint. Binary corners provide no
  // such evidence and must keep the established reconstruction.
  if((expected<16 || expected>240) && !(a>=16 && a<=240 && b>=16 && b<=240))return false;
  if(validate) {
    const int same=std::min(AA_FIELD_CAP,std::max(144,n/AA_FIELD_DIVISOR)),residual=std::min(1024,std::max(144,n/64));
    int fa=0,ba=0;uint32_t mid_fg=0,mid_bg=0;
    for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u) {
      const auto value=at(u,v);
      if((u && v) || u<0 || v<0) {
        const bool background=u>0 && v>0;
        if(distance(value,packed(background?bg:fg))>same)return false;
        const auto c=rgb(value);const int projection=((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2;
        if(projection>=16*n+n/2-continuation_margin && projection<=240*n+n/2+continuation_margin) {
          const int q=ratio(value);if(distance(value,blend(bg,fg,q))<=144)(background?mid_bg:mid_fg)|=1u<<((v+2)*5+u+2);
        }
        fa+=!background && projection>=(256-AA_ENDPOINT_TOLERANCE)*n && projection<(257+AA_ENDPOINT_TOLERANCE)*n;
        ba+=background && projection>=-AA_ENDPOINT_TOLERANCE*n && projection<(1+AA_ENDPOINT_TOLERANCE)*n;
      } else {
        const int q=ratio(value),target=u?a:v?b:expected;
        if(std::abs(q-target)>AA_PHASE_TOLERANCE || distance(value,blend(bg,fg,q))>residual)return false;
      }
    }
    if(fa<14 || ba<3 || ((repeated_intermediate(mid_fg) || repeated_intermediate(mid_bg))))return false;
  }
  out={packed(fg),packed(bg),a,b};return true;
}
void joint_render(const Joint &p,int orientation,int role,uint32_t *out) {
  const auto bg=rgb(p.bg),fg=rgb(p.fg);
  for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
    int xx=orientation&1?2-x:x,yy=orientation&2?2-y:y;
    const int qx=clip(p.b*3-xx*256,0,256),qy=clip(p.a*3-yy*256,0,256);
    const int coverage=role==11?256-((256-qx)*(256-qy)+128)/256:role<=13?qy:qx;
    out[y*3+x]=blend(bg,fg,coverage);
  }
}
int joint_role(const Joint &p,int k,int dx,int dy) {
  const int sx=k&1?-1:1,sy=k&2?-1:1;
  if(!dx && !dy)return 11;
  if(!dy && dx*sx>=1 && dx*sx<=2 && p.a>0 && p.a<256)return 11+dx*sx;
  if(!dx && dy*sy>=1 && dy*sy<=2 && p.b>0 && p.b<256)return 13+dy*sy;
  return std::abs(dx)<=2 && std::abs(dy)<=2?5:0;
}
// Endpoint/phase estimates belong to the output pixel. Shared straight
// arms therefore render identically regardless of which joint recognized
// them, and meet the standalone axial path without color or phase seams.
bool local_edge(const Frame &f,int x,int y,int nx,int ny,uint32_t *out) {
  // Canonical positive normal makes shared-edge rendering independent of
  // the orientation of the anchor which admitted it.
  nx=nx!=0;ny=ny!=0;
  const int tx=ny,ty=nx;
  auto at=[&](int u,int v){return pixel(f,x+u*tx+v*nx,y+u*ty+v*ny);};
  const auto fg=median3(rgb(at(-1,-1)),rgb(at(0,-1)),rgb(at(1,-1))),
             bg=median3(rgb(at(-1,1)),rgb(at(0,1)),rgb(at(1,1))),c=rgb(pixel(f,x,y));
  const int dr=fg.r-bg.r,dg=fg.g-bg.g,db=fg.b-bg.b,n=dr*dr+dg*dg+db*db;
  bool usable=n>=1800;int a=0;
  if(usable) {
    a=floor_div(((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2,n);
    usable=a>=0 && a<=256 && distance(packed(c),blend(bg,fg,a))<=std::min(1024,std::max(144,n/64));
  }
  if(!usable) {std::fill_n(out,9,packed(c));return false;}
  for(int yy=0;yy<3;++yy)for(int xx=0;xx<3;++xx) {
    const int i=nx?xx:yy;
    out[yy*3+xx]=blend(bg,fg,clip(3*a-i*256,0,256));
  }
  return true;
}
void claim_render(const Frame &f,int x,int y,const Joint &p,int orientation,int role,uint32_t *out) {
  const int dx=orientation&1?-1:1,dy=orientation&2?-1:1;
  if(role==5)std::fill_n(out,9,pixel(f,x,y));
  else if(role>=12)local_edge(f,x,y,role<=13?0:dx,role<=13?dy:0,out);
  else if(p.a==0)local_edge(f,x,y,dx,0,out);
  else if(p.b==0)local_edge(f,x,y,0,dy,out);
  else joint_render(p,orientation,role,out);
}
// A zero-coverage end immediately outside a recognized straight AA strip.
// Its own actual 5x5 source must prove background on the exterior side and
// behind the end, a continued foreground field, and two partial samples
// ahead. This joins the new strip to an older curved fallback without
// retaining a disconnected fallback tip in the proven exterior.
bool opaque_edge_cap(const Frame &f,int x,int y,int nx,int ny,int tx,int ty) {
  if(x<2 || y<2 || x>=f.w-2 || y>=f.h-2)return false;
  auto at=[&](int u,int v){return pixel(f,x+u*nx+v*tx,y+u*ny+v*ty);};
  const auto bp=at(0,0);if(at(-1,0)!=bp || at(0,-1)!=bp || at(0,-2)!=bp)return false;
  const auto fg=median3(rgb(at(1,0)),rgb(at(1,1)),rgb(at(1,2))),bg=rgb(bp);
  const auto fp=packed(fg);const int dr=fg.r-bg.r,dg=fg.g-bg.g,db=fg.b-bg.b,n=dr*dr+dg*dg+db*db;
  if(n<6912)return false;
  auto ratio=[&](uint32_t value){auto c=rgb(value);return floor_div(((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2,n);};
  const int a=ratio(at(0,1)),b=ratio(at(0,2));if(a<16 || a>240 || b<16 || b>240 || std::abs(a-b)>AA_PHASE_TOLERANCE)return false;
  for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u) {
    const int xx=x+u*nx+v*tx,yy=y+u*ny+v*ty;if(bool(f.text[size_t(yy)*f.w+xx])!=bool(f.text[size_t(y)*f.w+x]))return false;
    const auto value=at(u,v);
    if(u<0) {if(!near(value,bp))return false;}
    else if(u>0 && v>=0) {if(!near(value,fp))return false;}
    else {
      const int q=ratio(value);if(q<0 || q>256 || distance(value,blend(bg,fg,q))>144)return false;
      if(u>0 && q>ratio(at(u,v+1))+8)return false;
    }
  }
  return true;
}
struct Corner {
  int x, y, kind, dx, dy;
  uint32_t fg, bg;
  bool solid, field_gap, extended_endpoint;
};
// Copy an AA neighbor only when its own two-column source strip proves
// the straight continuation. An AA center does not prove four square blocks.
bool straight_corner_neighbor(const Frame &f,const Corner &p,int dx,int dy) {
  bool positive=(dx==p.dx || dy==p.dy);
  for(int i=1;i<=2;++i)for(int j=-1;j<=1;++j) {
    const int xx=p.x+i*dx+(dy?j*p.dx:0);
    const int yy=p.y+i*dy+(dx?j*p.dy:0);
    const auto expected=positive && j>=0?p.fg:p.bg;
    if(!near(pixel(f,xx,yy),expected)) return false;
  }
  return true;
}
struct BoldJoin {
  int x0, y0, x1, y1;
  int cap_x, cap_y;
};
struct JoinCells {
  std::array<std::array<int, 2>, 4> points;
  unsigned count = 0;
  int arms = 0;
};
bool masked_bold_join(const Frame &f, int x, int y, uint32_t center,
                      unsigned key, JoinCells &out) {
  // Same mixed-width square T geometry as bold_join. The cached core key
  // selects orientation; only relevant outer continuation cells are cared
  // about. This also accepts row shading without treating it as a gap.
  int mx=0,my=0,bar=0,stem=0,side=0;
  switch (key) {
  case 160: my=-1; bar=2; stem=1; break;
  case 5:   my=1; bar=2; stem=1; break;
  case 132: mx=-1; bar=2; stem=1; break;
  case 33:  mx=1; bar=2; stem=1; break;
  case 39:  my=-1; bar=1; stem=2; side=1; break;
  case 135: my=-1; bar=1; stem=2; side=-1; break;
  case 228: my=1; bar=1; stem=2; side=1; break;
  case 225: my=1; bar=1; stem=2; side=-1; break;
  case 169: mx=-1; bar=1; stem=2; side=1; break;
  case 45:  mx=-1; bar=1; stem=2; side=-1; break;
  case 149: mx=1; bar=1; stem=2; side=1; break;
  case 180: mx=1; bar=1; stem=2; side=-1; break;
  default: return false;
  }
  const int ux=-my,uy=mx;
  auto at=[&](int u,int v) {return pixel(f,x+u*ux+v*mx,y+u*uy+v*my);};
  const int left=stem==1?-1:(side==1?-1:-2),
            right=stem==1?1:(side==1?2:1);
  const auto bg=packed(mean(rgb(at(left,-1)),rgb(at(right,-1))));
  const int contrast=distance(center,bg);
  if (contrast<3600) return false;
  const int same=std::max(144,contrast/128),
            continuation=std::max(144,contrast/9),
            separated=std::max(144,contrast/16);
  if (distance(at(left,-1),bg)>same || distance(at(right,-1),bg)>same)
    return false;
  bool background_continues=false;
  for (int u : {left,right}) {
    auto c=at(u,-2);
    if (distance(c,center)<=separated) return false;
    background_continues |= distance(c,bg)<=same;
  }
  if (!background_continues) return false;
  const int field_tolerance=std::max(144,contrast/32);
  for (int v=bar;v<=2;++v) {
    auto c=at(0,v);
    // A field may return beyond a shadow, but an unrelated third tonal
    // stripe must not count as a missing arm. Require an observed gap layer.
    if (distance(c,center)<=separated ||
        (distance(c,bg)>field_tolerance && distance(c,at(left,-2))>field_tolerance &&
         distance(c,at(right,-2))>field_tolerance)) return false;
  }
  // Five actual pixels continue the bar. A two-pixel-wide bar needs its
  // second column/row only at the three central positions: far cap corners
  // may be clipped or belong to another arm, as in a small E.
  for (int u=-2;u<=2;++u) {
    const auto c=at(u,0);
    if (distance(c,center)>continuation) return false;
    if (bar==2 && std::abs(u)<=1 && distance(at(u,1),c)>same)
      return false;
  }
  for (int v : {-1,-2}) {
    const auto c=at(0,v);
    if (distance(c,center)>continuation) return false;
    if (stem==2 && (distance(at(side,v),c)>same ||
                   distance(at(side,v),center)>continuation))
      return false;
  }
  auto add=[&](int u,int v) {
    out.points[out.count++]={x+u*ux+v*mx,y+u*uy+v*my};
  };
  add(0,0);
  if (stem==2) add(side,0);
  add(left,-1); add(right,-1);
  out.arms=stem==2?3:4;
  return true;
}
bool cross_join(const Frame &f, int x, int y, uint32_t center,
                unsigned corner_mask,
                std::array<std::array<int, 2>, 8> &out) {
  // A straight two-pixel stem crossed by a one-pixel bar. The two
  // foreground diagonals identify the other stem column. Distant corners
  // are deliberately irrelevant: a serif/ascender can occupy those cells.
  int ux = 0, uy = 0;
  switch (corner_mask) {
  case 0x21: ux = 1; break;
  case 0x84: ux = -1; break;
  case 0x05: uy = 1; break;
  case 0xa0: uy = -1; break;
  default: return false;
  }
  const int vx = -uy, vy = ux;
  auto at = [&](int u, int v) {
    return pixel(f, x + u*ux + v*vx, y + u*uy + v*vy);
  };
  // Reject one-sided T joins before reading the longer stem witnesses.
  if (!near(at(-1, 0), center) || !near(at(2, 0), center))
    return false;
  const auto background = at(-1, 1);
  if (distance(center, background) < 1800)
    return false;
  for (int u : {-1, 2})
    for (int v : {-1, 1})
      if (!near(at(u, v), background))
        return false;
  for (int v = -2; v <= 2; ++v)
    for (int u : {0, 1})
      if (!near(at(u, v), center))
        return false;
  // A bar may continue or end at a flat cap. Exclude an immediate diagonal
  // continuation, a neighboring stroke or a third surface at either end.
  for (int u : {-2, 3}) {
    for (int v = -1; v <= 1; ++v) {
      int xx = x + u*ux + v*vx, yy = y + u*uy + v*vy;
      if (xx < 0 || yy < 0 || xx >= f.w || yy >= f.h)
        return false;
      auto c = at(u, v);
      if (v ? !near(c, background)
            : !(near(c, background) || near(c, center)))
        return false;
    }
  }
  constexpr int cells[8][2] = {{0,0},{1,0},{-1,-1},{-1,1},
                                {2,-1},{2,1},{-1,0},{2,0}};
  for (int i = 0; i < 8; ++i)
    out[i] = {x + cells[i][0]*ux + cells[i][1]*vx,
              y + cells[i][0]*uy + cells[i][1]*vy};
  return true;
}
bool bold_join(const Frame &f, int x, int y, uint32_t center, int arms,
               BoldJoin &out) {
  int diagonal = 0;
  for (int dy : {-1, 1})
    for (int dx : {-1, 1})
      diagonal += near(pixel(f, x + dx, y + dy), center);
  // These counts distinguish a one/two-pixel stem and one/two-pixel bar.
  // The remaining 5x5 check must prove the full two-surface T shape.
  if (!((arms == 3 && diagonal == 1) ||
        (arms == 4 && (diagonal == 2 || diagonal == 3))))
    return false;
  int missing = 0, mx = 0, my = 0;
  for (auto d : std::array<std::array<int, 2>, 4>{
           {{{0, 1}}, {{1, 0}}, {{0, -1}}, {{-1, 0}}}})
    if (!near(pixel(f, x + 2*d[0], y + 2*d[1]), center)) {
      ++missing;
      mx = d[0];
      my = d[1];
    }
  if (missing != 1)
    return false;
  const int ux = -my, uy = mx;
  const int stem = diagonal == 2 ? 1 : 2, bar = arms == 3 ? 1 : 2;
  int side = 1;
  const bool left = near(pixel(f, x - mx - ux, y - my - uy), center),
             right = near(pixel(f, x - mx + ux, y - my + uy), center);
  if (stem == 1 ? (left || right) : (left == right))
    return false;
  if (stem == 2)
    side = right ? 1 : -1;
  auto background = pixel(f, x - mx - side*ux, y - my - side*uy);
  if (distance(center, background) < 1800)
    return false;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      int u = dx*ux + dy*uy, v = dx*mx + dy*my;
      bool foreground = (v >= 0 && v < bar) ||
                        (v < 0 && (u == 0 || (stem == 2 && u == side)));
      if (!near(pixel(f, x + dx, y + dy), foreground ? center : background))
        return false;
    }
  // Square the two inside background corners; the witnessed cap below may
  // also claim a stroke endpoint. Later antialiased rules keep precedence.
  int opposite = stem == 2 ? 2*side : side;
  out = {x - mx - side*ux, y - my - side*uy,
         x - mx + opposite*ux, y - my + opposite*uy, -1, -1};
  // A one-pixel arm may end two pixels from this junction (the F midbar).
  // Its endpoint is square only when the source proves a complete flat cap:
  // coherent background ahead, on both sides, and at the two front corners.
  int qx = x - 2*mx, qy = y - 2*my;
  if (stem == 1 && qx > 0 && qy > 0 && qx < f.w - 1 && qy < f.h - 1 &&
      near(pixel(f, qx - ux, qy - uy), background) &&
      near(pixel(f, qx + ux, qy + uy), background) &&
      near(pixel(f, qx - mx, qy - my), background) &&
      near(pixel(f, qx - mx - ux, qy - my - uy), background) &&
      near(pixel(f, qx - mx + ux, qy - my + uy), background)) {
    out.cap_x = qx;
    out.cap_y = qy;
  }
  return true;
}
bool corner(const Frame &f, int x, int y, int kind, Corner &p) {
  int dy = kind <= 2 ? 1 : -1, dx = (kind == 1 || kind == 3) ? 1 : -1;
  auto h = pixel(f, x + dx, y), v = pixel(f, x, y + dy),
       oh = pixel(f, x - dx, y), ov = pixel(f, x, y - dy);
  const int hv=distance(h,v);
  if (hv>AA_ENDPOINT_CAP) return false;
  const int background_pair=distance(oh,ov);
  if (background_pair>AA_ENDPOINT_CAP) return false;
  const bool shaded_endpoint=hv>144 || background_pair>144;
  Color fg = mean(rgb(h), rgb(v)), bg = mean(rgb(oh), rgb(ov)),
        c = rgb(pixel(f, x, y));
  if (shaded_endpoint) {
    // The later q>=224/q<=32 witnesses and n>=6912 imply more than
    // 3456 squared contrast between these rounded near-pair means.
    // Reject flat/noisy fields before estimating twelve endpoint samples.
    if(distance(packed(fg),packed(bg))<3456) return false;
    if (x<2 || y<2 || x>=f.w-2 || y>=f.h-2) return false;
    fg=continued_endpoint(f,x,y,dx,dy);
    bg=continued_endpoint(f,x,y,-dx,-dy);
  }
  int dr = fg.r - bg.r, dg = fg.g - bg.g, db = fg.b - bg.b,
      n = dr * dr + dg * dg + db * db;
  if (n < 1800)
    return false;
  if (shaded_endpoint && (n<6912 || std::max(hv,background_pair)>
        std::min(AA_ENDPOINT_CAP,std::max(144,n/96)))) return false;
  int a = floor_div(
      ((c.r - bg.r) * dr + (c.g - bg.g) * dg + (c.b - bg.b) * db) * 256 + n / 2,
      n);
  const bool extended_endpoint = a < 48 || a > 232;
  // Faint cuts need two coherent surfaces and source continuation on each
  // side. A ratio alone cannot distinguish an AA tip from a tonal field.
  const int residual = distance(pixel(f,x,y),blend(bg,fg,a));
  const int allowance = std::min(1024,std::max(144,n/64));
  if (a < 16 || a > 248 || (extended_endpoint && n < 6912) || residual > allowance)
    return false;
  // Newly widened admissions need a majority of continued samples on each
  // estimated surface. Existing accepted cases keep their established path.
  const int old_pair=std::min(576,std::max(144,n/128));
  if ((shaded_endpoint && std::max(hv,background_pair)>old_pair) ||
      residual>std::min(576,std::max(144,n/64))) {
    auto support=[&](int sx,int sy,Color endpoint) {
      int count=0;const int points[6][2]={{2,0},{0,2},{1,1},{2,1},{1,2},{2,2}};
      for(auto &point:points)
        count+=near(pixel(f,x+point[0]*sx,y+point[1]*sy),packed(endpoint));
      return count;
    };
    if(support(dx,dy,fg)<4 || support(-dx,-dy,bg)<4)return false;
  }
  bool shaded_continuation=false;
  if (residual > 144 || shaded_endpoint) {
    // Bounded mixture tolerance: preserve old accepted cases verbatim.
    // Newly admitted mixtures need repeated endpoints and a continued edge.
    if (n < 6912 || x<2 || y<2 || x>=f.w-2 || y>=f.h-2) return false;
    if (shaded_endpoint) {
      // Cross-multiplied Q8 tests reject unsupported arms before25 divisions.
      // These inequalities exactly match clipped q>=224 / q<=32.
      const auto projection = [&](int xx,int yy) {
        auto c=rgb(pixel(f,x+xx,y+yy));
        return ((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2;
      };
      for(int step:{1,2})
        if(projection(step*dx,0)<224*n || projection(0,step*dy)<224*n ||
           projection(-step*dx,0)>=33*n || projection(0,-step*dy)>=33*n)
          return false;
    }
    int ratios[5][5], endpoints=0, na=0, nb=0;
    for (int yy=-2; yy<=2; ++yy) for (int xx=-2; xx<=2; ++xx) {
      const auto value=pixel(f,x+xx,y+yy); const auto cc=rgb(value);
      const int q=clip(floor_div(((cc.r-bg.r)*dr+(cc.g-bg.g)*dg+(cc.b-bg.b)*db)*256+n/2,n),0,256);
      if (distance(value,blend(bg,fg,q))>allowance) return false;
      ratios[yy+2][xx+2]=q;
      na+=q<=16; nb+=q>=240; endpoints+=(q<=16 || q>=240);
    }
    if (endpoints<18 || na<3 || nb<3) return false;
    shaded_continuation=shaded_endpoint;
    for (int yy=0;yy<5;++yy) for (int xx=0;xx<5;++xx) {
      if (xx+dx>=0 && xx+dx<5 && ratios[yy][xx+dx]<ratios[yy][xx]-16) return false;
      if (yy+dy>=0 && yy+dy<5 && ratios[yy+dy][xx]<ratios[yy][xx]-16) return false;
    }
  }
  const bool near_solid = a > 220;
  const bool endpoint_corner = near_solid || a < 48;
  const bool solid = shaded_continuation || near(pixel(f, x + 2 * dx, y), packed(fg)) ||
                     near(pixel(f, x, y + 2 * dy), packed(fg));
  if (extended_endpoint) {
    const bool background_solid = shaded_continuation ||
        near(pixel(f, x - 2 * dx, y), packed(bg)) ||
        near(pixel(f, x, y - 2 * dy), packed(bg));
    if (!solid || !background_solid)
      return false;
  }
  // A one-pixel outline/shadow can surround a hole in the underlying field.
  // Its field color can accidentally fit the foreground/shadow mixture. Two
  // continued foreground arms and the field's exact return beyond the shadow
  // distinguish this supported three-surface gap from a two-surface AA tip.
  const auto center = pixel(f, x, y);
  const bool field_gap = !near_solid &&
      !near(center, packed(fg)) && !near(center, packed(bg)) &&
      near(pixel(f, x + 2 * dx, y), packed(fg)) &&
      near(pixel(f, x, y + 2 * dy), packed(fg)) &&
      pixel(f, x - 2 * dx, y) == center &&
      pixel(f, x, y - 2 * dy) == center &&
      pixel(f, x - 2 * dx, y - 2 * dy) == center;
  if (n < 6912 || endpoint_corner) {
    // A tiny clipped tip can be AA, but a smooth tonal ramp can mimic it.
    // Near-solid and newly admitted weak cases need color-segment coherence
    // and mostly actual endpoint samples; other ratios retain old admission.
    int good = 0, endpoints = 0;
    const auto foreground = packed(fg), background = packed(bg);
    for (int yy = -2; yy <= 2; ++yy)
      for (int xx = -2; xx <= 2; ++xx) {
        auto c = rgb(pixel(f, x + xx, y + yy));
        int q = clip(floor_div(((c.r - bg.r) * dr + (c.g - bg.g) * dg +
                                (c.b - bg.b) * db) *
                                       256 +
                                   n / 2,
                               n),
                     0, 256);
        auto value = pixel(f, x + xx, y + yy);
        good += near(value, blend(bg, fg, q));
        if (endpoint_corner)
          endpoints += near(value, foreground) || near(value, background);
      }
    if (good < 23 || (endpoint_corner && endpoints < 20))
      return false;
  }
  p = {x,
       y,
       kind,
       dx,
       dy,
       packed(fg),
       packed(bg),
       solid,
       field_gap,
       extended_endpoint};
  return true;
}
// A T or + is an orthogonal junction, including a one-pixel short arm.
// Use two coherent surfaces (with contrast-relative row shading), require
// square background quadrants, and veto a short arm that turns diagonally.
// These are geometric witnesses, independent of caller text rectangles.
bool general_junction(const Frame &f,int x,int y) {
  const auto fg=pixel(f,x,y),bg=pixel(f,x-1,y-1);
  const int contrast=distance(fg,bg);
  if(contrast<3600)return false;
  const int same=std::max(144,contrast/64),field=std::max(144,contrast/16);
  auto on=[&](uint32_t c){return distance(c,fg)<=same;};
  auto off=[&](uint32_t c){return distance(c,bg)<=field;};
  for(int dy:{-1,1})for(int dx:{-1,1})if(!off(pixel(f,x+dx,y+dy)))return false;
  int arms=0;
  for(auto d:std::array<std::array<int,2>,4>{{{{0,1}},{{1,0}},{{0,-1}},{{-1,0}}}}) {
    const int dx=d[0],dy=d[1];const auto v=pixel(f,x+dx,y+dy);
    const auto next=pixel(f,x+2*dx,y+2*dy);
    if(on(v)) {
      ++arms;
      if(!on(next)) {
        if(!off(next))return false;
        for(int sign:{-1,1})
          if(!off(pixel(f,x+2*dx-sign*dy,y+2*dy+sign*dx)))return false;
      }
    }else if(!off(v) || !off(next))return false;
  }
  if(arms<3)return false;
  // Far diagonal samples may contain other strokes, but no third surface:
  // do not turn texture or a partial AA curve into a binary junction.
  for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
    const auto v=pixel(f,x+dx,y+dy);if(!on(v) && !off(v))return false;
  }
  return true;
}
// A source-exact3x3 T/+ can sit directly below a panel bevel. The bevel
// is outside the junction and must not force its bar back to HQ rounding.
// Require a real continued arm and reject diagonal returns at short arms;
// copy only the witnessed3x3 cells, never borrow the outer surface colour.
bool local_square_junction(const Frame &f,int x,int y) {
  const auto fg=pixel(f,x,y),bg=pixel(f,x-1,y-1);
  if(distance(fg,bg)<6912)return false;
  for(int dy:{-1,1})for(int dx:{-1,1})if(pixel(f,x+dx,y+dy)!=bg)return false;
  int arms=0,continued=0;
  for(auto d:std::array<std::array<int,2>,4>{{{{0,1}},{{1,0}},{{0,-1}},{{-1,0}}}}) {
    const int dx=d[0],dy=d[1];const auto c=pixel(f,x+dx,y+dy);
    if(c!=fg && c!=bg)return false;
    const auto next=pixel(f,x+2*dx,y+2*dy);
    if(c==fg) {
      ++arms;
      if(near(next,fg))++continued;
      else for(int sign:{-1,1})
        if(near(pixel(f,x+2*dx-sign*dy,y+2*dy+sign*dx),fg))return false;
    }else if(near(next,fg))return false;
  }
  return arms>=3 && continued>=1;
}
// Wider +: infer complete straight rows and columns, then verify their
// rectangular union against all 25 source samples. No diagonal X or filled
// rectangle can supply the required one/two-cell-wide crossing and fields.
bool general_wide_cross(const Frame &f,int x,int y) {
  const auto fg=pixel(f,x,y);uint32_t bg=fg;int separation=0;
  for(int dy:{-2,2})for(int dx:{-2,2}) {
    const auto c=pixel(f,x+dx,y+dy);const int n=distance(c,fg);
    if(n>separation){separation=n;bg=c;}
  }
  if(separation<3600)return false;
  const int same=std::max(144,separation/64),field=std::max(144,separation/16);
  bool rows[5]={},cols[5]={};int nr=0,nc=0;
  for(int i=-2;i<=2;++i) {
    bool row=true,col=true;
    for(int j=-2;j<=2;++j) {
      row &= distance(pixel(f,x+j,y+i),fg)<=same;
      col &= distance(pixel(f,x+i,y+j),fg)<=same;
    }
    rows[i+2]=row;cols[i+2]=col;nr+=row;nc+=col;
  }
  if(nr<1 || nr>2 || nc<1 || nc>2 || !rows[2] || !cols[2] ||
     rows[0] || rows[4] || cols[0] || cols[4])return false;
  for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
    const auto c=pixel(f,x+dx,y+dy);
    if(rows[dy+2] || cols[dx+2]) {if(distance(c,fg)>same)return false;}
    else if(distance(c,bg)>field)return false;
  }
  return true;
}
bool junction_key(unsigned key) {
  switch(key) {
  // Thin T/+ and one/two-pixel bar/stem variants. Cheap admission only;
  // actual RGB and radius-two continuation witnesses still decide.
  case 0xa7:case 0xb5:case 0xe5:case 0xad:case 0xa5:
  case 1:case 4:case 32:case 128:
  case 160:case 5:case 132:case 33:case 39:case 135:
  case 228:case 225:case 169:case 45:case 149:case 180:
    return true;
  default:return false;
  }
}
// Cheap 3x3 endpoint-repeat test before any full 5x5 fitting. Texture
// usually lacks two repeated color groups. A concave union is the exception:
// its core can expose one background corner and three foreground corners.
bool general_candidate(const Frame &f,int x,int y) {
  uint32_t c[9];f.core(x,y,c);
  const auto ends=std::minmax_element(c,c+9);
  if(distance(*ends.first,*ends.second)<3456)return false;
  int lo=0,hi=0;
  for(auto v:c){lo+=near(v,*ends.first);hi+=near(v,*ends.second);}
  const bool partial=c[4]!=*ends.first && c[4]!=*ends.second;
  if(partial && ((lo>=2 && hi>=2) ||
      (near(c[2],c[4]) && near(c[6],c[4])) ||
      (near(c[0],c[4]) && near(c[8],c[4]))))return true;
  constexpr int diagonal[]={0,2,8,6};
  for(int i=0;i<4;++i) {
    const auto bg=c[diagonal[i]],fg=c[diagonal[(i+1)&3]];
    if(distance(bg,fg)>=3456 && near(fg,c[diagonal[(i+2)&3]]) &&
       near(fg,c[diagonal[(i+3)&3]]))return true;
  }
  return false;
}
// No-hint admission is stricter than the authored-text path: two continued
// endpoint fields, actual fractional coverage, a coherent 5x5 RGB segment
// and monotone coverage in the 3x3 core. No font-specific neighboring letter
// can supply missing evidence. Binary/shadow notches stay in the hinted path.
bool general_corner(const Frame &f,const Corner &p) {
  if(!p.solid || p.field_gap)return false;
  const auto fg=rgb(p.fg),bg=rgb(p.bg);
  const int dr=fg.r-bg.r,dg=fg.g-bg.g,db=fg.b-bg.b,n=dr*dr+dg*dg+db*db;
  if(n<6912)return false;
  for(int step:{1})
    if(!near(pixel(f,p.x+step*p.dx,p.y),p.fg) ||
       !near(pixel(f,p.x,p.y+step*p.dy),p.fg) ||
       !near(pixel(f,p.x-step*p.dx,p.y),p.bg) ||
       !near(pixel(f,p.x,p.y-step*p.dy),p.bg))return false;
  int q[5][5],na=0,nb=0;
  constexpr int radius=DL_GENERAL_AA_SUPPORT/2;
  for(int y=-radius;y<=radius;++y)for(int x=-radius;x<=radius;++x) {
    const auto value=pixel(f,p.x+x,p.y+y);const auto c=rgb(value);
    const int a=floor_div(((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2,n);
    if(a< -8 || a>264 || distance(value,blend(bg,fg,clip(a,0,256)))>144)return false;
    q[y+2][x+2]=clip(a,0,256);na+=a<=16;nb+=a>=240;
  }
  if(na<(radius==2?3:2) || nb<(radius==2?3:2) || na+nb<(radius==2?18:6))return false;
  // Continuation can curve back at radius two (a small icon or an O).
  // Enforce the local corner ordering in the core, while all 25 samples
  // must still obey the same RGB mixture segment and endpoint majority.
  constexpr int low=2-DL_GENERAL_MONOTONE_RADIUS,high=2+DL_GENERAL_MONOTONE_RADIUS;
  for(int y=low;y<=high;++y)for(int x=low;x<=high;++x) {
    if(x+p.dx>=low && x+p.dx<=high && q[y][x+p.dx]<q[y][x]-16)return false;
    if(y+p.dy>=low && y+p.dy<=high && q[y+p.dy][x]<q[y][x]-16)return false;
  }
  return true;
}
// A thin curved elbow is not a monotone filled quadrant: both axial
// stroke arms can be foreground while every adjacent diagonal is background.
// A 5x5 single-edge fit rejects real O/P/S bitmap corners when another stroke
// returns inside that window. Admit only this measured 3x3 two-surface case;
// it is not a blanket relaxation for arbitrary corners, gradients or texture.
bool general_thin_corner(const Frame &f,const Corner &p) {
  if(!p.solid || p.field_gap || p.extended_endpoint)return false;
  const auto fg=rgb(p.fg),bg=rgb(p.bg),c=rgb(pixel(f,p.x,p.y));
  const int dr=fg.r-bg.r,dg=fg.g-bg.g,db=fg.b-bg.b;
  const int n=dr*dr+dg*dg+db*db;
  if(n<6912)return false;
  const int a=floor_div(((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2,n);
  if(a<48 || a>220 || distance(pixel(f,p.x,p.y),blend(bg,fg,a))>144)return false;
  if(!near(pixel(f,p.x+p.dx,p.y),p.fg) ||
     !near(pixel(f,p.x,p.y+p.dy),p.fg) ||
     !near(pixel(f,p.x-p.dx,p.y),p.bg) ||
     !near(pixel(f,p.x,p.y-p.dy),p.bg))return false;
  for(int dy:{-1,1})for(int dx:{-1,1})
    if(!near(pixel(f,p.x+dx,p.y+dy),p.bg))return false;
  // At least one actual radius-two arm proves continuation. Other outer
  // samples can belong to the returning curve or a neighbouring surface.
  return near(pixel(f,p.x+2*p.dx,p.y),p.fg) ||
         near(pixel(f,p.x,p.y+2*p.dy),p.fg);
}
// Cheap admission for a fractional corner whose two low axial samples
// may themselves be mixtures. Random colour texture rarely has both pairs.
bool mixed_corner_key(unsigned key) {
  const unsigned card=key&0x5au;
  return card==0x5a || card==0x0a || card==0x12 || card==0x48 || card==0x50 ||
         card==0x58 || card==0x52 || card==0x4a || card==0x1a;
}
bool mixed_corner_candidate(const Frame &f,int x,int y) {
  for(int k=1;k<=4;++k) {
    const int dx=k==1 || k==3?1:-1,dy=k<=2?1:-1;
    const auto a=pixel(f,x+dx,y),b=pixel(f,x,y+dy),
               c=pixel(f,x-dx,y),d=pixel(f,x,y-dy);
    if(!near(a,b) || distance(c,d)>AA_ENDPOINT_CAP || distance(a,c)<6912)continue;
    return true;
  }
  const auto center=pixel(f,x,y);
  for(auto d:std::array<std::array<int,2>,4>{{{{0,1}},{{1,0}},{{0,-1}},{{-1,0}}}})
    if(near(pixel(f,x+d[0],y+d[1]),center) &&
       distance(pixel(f,x-d[0],y-d[1]),center)>=6912)return true;
  return false;
}
// A two-surface elbow may contain several measured intermediate shades.
// Choose observed endpoints, validate their shared RGB line over the actual
// 5x5, then classify geometry by coverage instead of treating each low arm as
// a separate background colour. No screenshot palette/letter is hardcoded.
bool mixed_corner(const Frame &f,int x,int y,int kind,AaTie &out) {
  const bool cap=kind>=5;
  constexpr int capx[]={1,0,-1,0},capy[]={0,1,0,-1};
  const int dx=cap?capx[kind-5]:(kind==1 || kind==3?1:-1),
            dy=cap?capy[kind-5]:(kind<=2?1:-1);
  const auto first=cap?pixel(f,x,y):pixel(f,x+dx,y),
             second=cap?pixel(f,x+dx,y+dy):pixel(f,x,y+dy);
  if(!near(first,second))return false;
  const auto fg=packed(mean(rgb(first),rgb(second)));
  Color low{};int low_count=0;
  if(cap) {
    for(auto d:std::array<std::array<int,2>,4>{{{{0,1}},{{1,0}},{{0,-1}},{{-1,0}}}})
      if(d[0]!=dx || d[1]!=dy) {
        const auto c=rgb(pixel(f,x+d[0],y+d[1]));
        low.r+=c.r;low.g+=c.g;low.b+=c.b;++low_count;
      }
  }else {
    const auto a=rgb(pixel(f,x-dx,y)),b=rgb(pixel(f,x,y-dy));
    low={a.r+b.r,a.g+b.g,a.b+b.b};low_count=2;
  }
  const auto low_color=packed({(low.r+low_count/2)/low_count,
                              (low.g+low_count/2)/low_count,
                              (low.b+low_count/2)/low_count});
  // Every accepted diagonal has coverage<=64 while the high arms>=240.
  // It must be nearer the observed low arms than the high endpoint. Reject
  // the opposite interpretation using only the core before any 5x5 fit.
  for(int sy:{-1,1})for(int sx:{-1,1}) {
    const auto c=pixel(f,x+sx,y+sy);
    if(distance(c,fg)<=distance(c,low_color))return false;
  }
  uint32_t bg=fg;int n=0;
  for(int yy=-2;yy<=2;++yy)for(int xx=-2;xx<=2;++xx) {
    const auto c=pixel(f,x+xx,y+yy);const int d=distance(c,fg);
    if(d>n || (d==n && c<bg)){n=d;bg=c;}
  }
  if(n<6912)return false;
  const auto fc=rgb(fg),bc=rgb(bg);
  const int dr=fc.r-bc.r,dg=fc.g-bc.g,db=fc.b-bc.b;
  int q[5][5],fg_count=0,bg_count=0;
  for(int yy=-2;yy<=2;++yy)for(int xx=-2;xx<=2;++xx) {
    const auto value=pixel(f,x+xx,y+yy);int a;
    if(value==fg)a=256;
    else if(value==bg)a=0;
    else {
      const auto c=rgb(value);
      a=floor_div(((c.r-bc.r)*dr+(c.g-bc.g)*dg+(c.b-bc.b)*db)*256+n/2,n);
      if(a< -8 || a>264 || distance(value,blend(bc,fc,clip(a,0,256)))>144)return false;
    }
    q[yy+2][xx+2]=clip(a,0,256);fg_count+=near(value,fg);bg_count+=near(value,bg);
  }
  const bool opaque=q[2][2]>=240;
  if(fg_count<3 || bg_count<2 || (!opaque && (q[2][2]<48 || q[2][2]>220)))return false;
  if(cap) {
    if(!opaque || q[2+dy][2+dx]<240 || q[2+2*dy][2+2*dx]<240 ||
       q[2-2*dy][2-2*dx]>64)return false;
    for(auto d:std::array<std::array<int,2>,4>{{{{0,1}},{{1,0}},{{0,-1}},{{-1,0}}}})
      if((d[0]!=dx || d[1]!=dy) && q[2+d[1]][2+d[0]]>64)return false;
  }else {
    if(q[2][2+dx]<240 || q[2+dy][2]<240 ||
       q[2][2-dx]>64 || q[2-dy][2]>64)return false;
    if(opaque ? (q[2][2+2*dx]<240 || q[2+2*dy][2]<240)
              : (q[2][2+2*dx]<240 && q[2+2*dy][2]<240))return false;
  }
  for(int sy:{-1,1})for(int sx:{-1,1})if(q[2+sy][2+sx]>64)return false;
  // Do not substitute for the old measured thin elbow unless it exposes
  // an actual partial low arm/diagonal, the missing case in shaded fonts.
  bool partial=false;
  for(int yy=-1;yy<=1;++yy)for(int xx=-1;xx<=1;++xx)
    if((xx || yy) && q[yy+2][xx+2]>=16 && q[yy+2][xx+2]<=64)partial=true;
  if(!partial)return false;
  out.index=size_t(y)*f.w+x;
  if(opaque) {out.pixels.fill(pixel(f,x,y));return true;}
  uint32_t canonical[9];f.core(x,y,canonical);
  canonical[(dy+1)*3+1]=canonical[3+dx+1]=fg;
  canonical[(-dy+1)*3+1]=canonical[3-dx+1]=bg;
  return aa_block(kind,canonical,out.pixels.data(),true);
}
// Local palette/coverage reconstruction. Three/four actual source colours
// are cheap to classify, unlike arbitrary RGB texture. The old robust AA
// routes remain available for larger/noisy palettes.
bool coverage_core(const Frame &f,int x,int y) {
  uint32_t colors[4],lo=0xffffff,hi=0;int count=0;
  for(int yy=-1;yy<=1;++yy)for(int xx=-1;xx<=1;++xx) {
    const auto c=pixel(f,x+xx,y+yy);int i=0;for(;i<count && colors[i]!=c;++i){}
    if(i==count){if(count==4)return false;colors[count++]=c;lo=std::min(lo,c);hi=std::max(hi,c);}
  }
  if(count<3 || distance(lo,hi)<6912)return false;
  const auto own=pixel(f,x,y);
  if(own!=lo && own!=hi)return true;
  // An endpoint can only use the new opaque-elbow case. Reject it before
  // collecting25samples unless two adjacent arms genuinely continue. An
  // outer fourth colour alone must not reinterpret a flat core endpoint.
  for(int dy:{-1,1})for(int dx:{-1,1})
    if(distance(own,pixel(f,x+dx,y))<=1024 && distance(own,pixel(f,x,y+dy))<=1024 &&
       distance(own,pixel(f,x+2*dx,y))<=1024 && distance(own,pixel(f,x,y+2*dy))<=1024 &&
       distance(own,pixel(f,x-dx,y))>=3000 && distance(own,pixel(f,x,y-dy))>=3000)return true;
  return false;
}
bool coverage_block(const Frame &f,int x,int y,AaTie &out) {
  uint32_t colors[4],lo=0xffffff,hi=0;int counts[4]={},count=0,ids[25];
  for(int yy=-2;yy<=2;++yy)for(int xx=-2;xx<=2;++xx) {
    const auto c=pixel(f,x+xx,y+yy);int i=0;for(;i<count && colors[i]!=c;++i){}
    if(i==count){if(count==4)return false;colors[count++]=c;lo=std::min(lo,c);hi=std::max(hi,c);}
    ++counts[i];ids[(yy+2)*5+xx+2]=i;
  }
  const auto lc=rgb(lo),hc=rgb(hi);const int dr=hc.r-lc.r,dg=hc.g-lc.g,db=hc.b-lc.b,n=dr*dr+dg*dg+db*db;
  if(count<3 || n<6912)return false;
  int values[4],nlo=0,nhi=0;bool intermediate=false;
  for(int i=0;i<count;++i) {
    int q;
    if(colors[i]==lo){q=0;nlo=counts[i];}
    else if(colors[i]==hi){q=256;nhi=counts[i];}
    else {
      const auto c=rgb(colors[i]);q=floor_div(((c.r-lc.r)*dr+(c.g-lc.g)*dg+(c.b-lc.b)*db)*256+n/2,n);
      if(q< -8 || q>264 || distance(colors[i],blend(lc,hc,clip(q,0,256)))>36)return false;
    }
    values[i]=clip(q,0,256);intermediate |=q>=16 && q<=240;
  }
  if(nlo<2 || nhi<2 || !intermediate)return false;
  int q[5][5];for(int i=0;i<25;++i)q[i/5][i%5]=values[ids[i]];
  const auto own=pixel(f,x,y);out.index=size_t(y)*f.w+x;
  // Preserve only a witnessed opaque convex elbow adjoining actual AA.
  // Generic endpoint copying would square unrelated material contours.
  if(own==lo || own==hi) {
    const bool invert=own==lo;auto at=[&](int u,int v){int a=q[2+v][2+u];return invert?256-a:a;};
    bool partial=false;
    for(int v=-1;v<=1;++v)for(int u=-1;u<=1;++u)if(u || v)partial |=at(u,v)>=48 && at(u,v)<=220;
    if(!partial)return false;
    for(int dy:{-1,1})for(int dx:{-1,1}) {
      bool sides=true;for(int sy:{-1,1})for(int sx:{-1,1})if(sx!=dx || sy!=dy)sides &=at(sx,sy)<=64;
      if(sides && at(dx,0)>=240 && at(0,dy)>=240 && at(-dx,0)<=64 && at(0,-dy)<=64 &&
         at(2*dx,0)>=240 && at(0,2*dy)>=240) {out.pixels.fill(own);return true;}
    }
    return false;
  }
  const int alpha=q[2][2];if(alpha<16 || alpha>240)return false;
  bool found=false,conflict=false;std::array<uint32_t,9> chosen{};
  auto offer=[&](const std::array<uint32_t,9> &pixels){if(found && chosen!=pixels)conflict=true;else {chosen=pixels;found=true;}};
  // Straight edges keep the existing whole5x5 plane validation. A local
  // three-pixel band alone also describes UI bevels/material boundaries;
  // treating those as fractional strips produces unsupported extra edges.
  // Convex cuts may border a filled quadrant or a thin returning stroke.
  // A weak diagonal fringe instead meets a real foreground diagonal plus
  // two repeated partial arms. Both redistribute only this pixel's area.
  for(bool invert:{false,true})for(int kind=1;kind<=4;++kind) {
    const int dx=kind==1 || kind==3?1:-1,dy=kind<=2?1:-1;
    auto at=[&](int u,int v){const int a=q[2+v][2+u];return invert?256-a:a;};
    const int a=at(0,0);if(a<16 || a>240 || (invert?nlo:nhi)<3)continue;
    bool sides=true;
    for(int sy:{-1,1})for(int sx:{-1,1})if(sx!=dx || sy!=dy)sides &= at(sx,sy)<=64;
    const bool cut=sides && at(dx,0)>=240 && at(0,dy)>=240 && at(-dx,0)<=64 && at(0,-dy)<=64 &&
        (at(2*dx,0)>=240 || at(0,2*dy)>=240);
    const bool fringe=sides && a<=128 && at(dx,dy)>=240 &&
        at(dx,0)>=16 && at(dx,0)<=128 && at(0,dy)>=16 && at(0,dy)<=128 &&
        std::abs(at(dx,0)-a)<=12 && std::abs(at(0,dy)-a)<=12 &&
        at(-dx,0)<=16 && at(0,-dy)<=16 &&
        (at(2*dx,dy)>=240 || at(dx,2*dy)>=240);
    if(!cut && !fringe)continue;
    uint32_t core[9];f.core(x,y,core);const uint32_t fg=invert?lo:hi,bg=invert?hi:lo;
    core[4+dx]=core[4+3*dy]=fg;core[4-dx]=core[4-3*dy]=bg;
    std::array<uint32_t,9> pixels{};if(aa_block(kind,core,pixels.data(),true))offer(pixels);
  }
  if(!found || conflict)return false;out.pixels=chosen;return true;
}
// A measured concave tip: three foreground arms around an AA centre,
// four background quadrants and an open background face. This is source
// geometry, independent of glyph names/hints. Only this source cell changes.
bool general_notch_key(unsigned key) {
  const unsigned d=key&0xa5u,c=key&0x5au;
  if(d && d!=0xa5u)return false;
  return c==2 || c==8 || c==16 || c==64 || c==0x1a || c==0x4a || c==0x52 || c==0x58 || c==0x5a;
}
const auto &notch_coverage_table() {
  static const auto table=[] {
    std::array<std::array<uint16_t,9>,257> t{};
    // Twice the area below X+Y=phase in a256-by-height rectangle.
    auto area2=[](int phase,int height) {
      const auto sq=[](int v){return v>0?v*v:0;};
      return sq(phase)-sq(phase-256)-sq(phase-height)+sq(phase-256-height);
    };
    auto render=[&](int phase) {
      std::array<uint16_t,9> a{};
      for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
        const int area=y==1?2*area2(phase+128-x*256,128):area2(phase+384-x*256,256);
        a[y*3+x]=uint16_t(clip((area+256)/512,0,256));
      }
      return a;
    };
    auto sum=[](const auto&a){return std::accumulate(a.begin(),a.end(),0);};
    for(int q=0;q<=256;++q) {
      int lo=-384,hi=768;
      while(lo<hi){const int mid=lo+(hi-lo)/2;if(sum(render(mid))<q*9)lo=mid+1;else hi=mid;}
      auto a=render(lo),b=render(lo-1);
      t[q]=std::abs(sum(b)-q*9)<=std::abs(sum(a)-q*9)?b:a;
    }
    return t;
  }();
  return table;
}
bool general_notch(const Frame &f,int x,int y,AaTie &out) {
  uint32_t core[9];f.core(x,y,core);
  if(!near(core[0],core[2]) || !near(core[0],core[6]) || !near(core[0],core[8]))return false;
  const auto d0=rgb(core[0]),d1=rgb(core[2]),d2=rgb(core[6]),d3=rgb(core[8]);
  const Color bg{(d0.r+d1.r+d2.r+d3.r+2)/4,(d0.g+d1.g+d2.g+d3.g+2)/4,(d0.b+d1.b+d2.b+d3.b+2)/4};
  bool found=false;std::array<uint32_t,9> chosen{};
  for(int rotation=0;rotation<4;++rotation) {
    auto point=[&](int u,int v){for(int i=0;i<rotation;++i){int a=u;u=-v;v=a;}return std::array<int,2>{u,v};};
    auto at=[&](int u,int v){const auto p=point(u,v);return pixel(f,x+p[0],y+p[1]);};
    const auto back=at(-1,0),upper=at(0,-1),lower=at(0,1);
    if(!near(back,upper) || !near(back,lower) || !near(at(1,0),packed(bg)) || !near(back,at(-2,0)))continue;
    const auto b=rgb(back),u=rgb(upper),v=rgb(lower);
    const Color fg{(b.r+u.r+v.r+1)/3,(b.g+u.g+v.g+1)/3,(b.b+u.b+v.b+1)/3};
    const int dr=fg.r-bg.r,dg=fg.g-bg.g,db=fg.b-bg.b,n=dr*dr+dg*dg+db*db;
    if(n<6912)continue;
    const auto c=rgb(core[4]);const int q=floor_div(((c.r-bg.r)*dr+(c.g-bg.g)*dg+(c.b-bg.b)*db)*256+n/2,n);
    if(q<96 || q>220 || distance(core[4],blend(bg,fg,q))>36)continue;
    std::array<uint32_t,9> pixels{};const auto &coverage=notch_coverage_table()[q];
    for(int yy=0;yy<3;++yy)for(int xx=0;xx<3;++xx){const auto p=point(xx-1,yy-1);pixels[(p[1]+1)*3+p[0]+1]=blend(bg,fg,coverage[yy*3+xx]);}
    if(found && chosen!=pixels)return false;chosen=pixels;found=true;
  }
  if(!found)return false;out={size_t(y)*f.w+x,chosen};return true;
}
struct Wedge {
  Color fg, bg;
  int alpha;
  bool shaded;
};
template<int Rotation>
uint32_t wedge_pixel(const Frame &f, int x, int y, int u, int v) {
  if constexpr (Rotation == 0) return pixel(f,x+u,y+v);
  if constexpr (Rotation == 1) return pixel(f,x-v,y+u);
  if constexpr (Rotation == 2) return pixel(f,x-u,y-v);
  return pixel(f,x+v,y-u);
}
template<int Rotation = 0>
bool wedge(const Frame &f, int x, int y, Wedge &p) {
  auto at = [&](int u,int v) {return wedge_pixel<Rotation>(f,x,y,u,v);};
  auto fg = at(-1,0), u = at(-1,-1), v = at(-1,1);
  auto bg = mean(rgb(u), rgb(v));
  auto c = rgb(at(0,0)), a = rgb(fg);
  int dr = a.r - bg.r, dg = a.g - bg.g, db = a.b - bg.b,
      n = dr * dr + dg * dg + db * db;
  const int luma = 77 * dr + 150 * dg + 29 * db;
  if (n <= 3600 || std::abs(luma) <= 20 * 256 ||
      !near(fg, at(-2,0)) || !near(u, v) ||
      distance(at(1,0), fg) <= 1225)
    return false;
  if constexpr (Rotation != 0) {
    // A rotated notch needs the complete thin fork, not merely three arms.
    // The two diagonal branches and two-pixel stem must be observed, with
    // the same background separating them. The far-front outer column is
    // outside this contour and may be a different panel/material.
    const auto background = packed(bg);
    for (int vv=-2;vv<=2;++vv)
      for (int uu=-2;uu<=1;++uu) {
        if (!uu && !vv) continue; // Existing mixture-ratio check below.
        const bool ink = (!vv && uu<0) ||
                         (uu==0 && std::abs(vv)==1) ||
                         (uu==1 && std::abs(vv)==2);
        if (!near(at(uu,vv),ink?fg:background)) return false;
      }
  }
  const auto upper = at(0,-1), lower = at(0,1);
  const int upper_delta = distance(fg, upper), lower_delta = distance(fg, lower);
  const bool shaded = upper_delta > 144 || lower_delta > 144;
  if (shaded) {
    // A shaded diagonal arm must continue with the same row-to-row ramp.
    // Coherent shadow witnesses distinguish it from unrelated color layers.
    const auto hi = rgb(upper), lo = rgb(lower);
    auto extrapolate = [&](Color row) {
      return packed({clip(2*row.r-a.r),clip(2*row.g-a.g),clip(2*row.b-a.b)});
    };
    if (upper_delta>n/9 || lower_delta>n/9 ||
        !near(packed(mean(hi,lo)),fg) ||
        !near(at(1,-2),extrapolate(hi)) ||
        !near(at(1,2),extrapolate(lo)) ||
        !near(at(-1,-2),u) || !near(at(-1,2),v))
      return false;
  }
  int q = floor_div(
      ((c.r - bg.r) * dr + (c.g - bg.g) * dg + (c.b - bg.b) * db) * 256 + n / 2,
      n);
  if (q < -13 || q >= 220 || distance(at(0,0), blend(bg, a, q)) >= 225)
    return false;
  // Extending the existing bright-text rule to dark strokes needs a coherent
  // color layer. Otherwise a black shadow beneath a differently colored glyph
  // can look like a notch while belonging to two overlapping outlines.
  if (luma < 0) {
    int foreign = 0;
    const auto background = packed(bg);
    for (int yy = -2; yy <= 2; ++yy)
      for (int xx = -2; xx <= 2; ++xx) {
        auto v = pixel(f, x + xx, y + yy);
        if (near(v, fg) || near(v, background))
          continue;
        auto c = rgb(v);
        int t = clip(floor_div(((c.r - bg.r) * dr + (c.g - bg.g) * dg +
                                (c.b - bg.b) * db) *
                                       256 +
                                   n / 2,
                               n),
                     0, 256);
        if (!near(v, blend(bg, a, t)) && ++foreign > 2)
          return false;
      }
  }
  p = {a, bg, clip(q, 0, 219), shaded};
  return true;
}
const auto &wedge_table() {
  static const auto table = [] {
    std::array<std::array<uint16_t, 21>, 220> t{};
    for (int a = 0; a < 220; ++a) {
      int depth = 768 - (a * 9 + 4) / 8;
      for (int y = 0; y < 7; ++y)
        for (int x = 0; x < 3; ++x) {
          int cut = 0;
          for (int yy = 0; yy < 16; ++yy)
            for (int xx = 0; xx < 16; ++xx) {
              int px = x * 256 + (2 * xx + 1) * 8,
                  py = (y - 2) * 256 + (2 * yy + 1) * 8;
              cut += std::abs(py - 384) <= px - (768 - depth);
            }
          t[a][y * 3 + x] = uint16_t(
              clip(128 + floor_div((128 - cut) * 320 + 128, 256), 0, 256));
        }
    }
    return t;
  }();
  return table;
}
} // namespace
void warm_stencils() {
  const uint32_t core[9] = {0,        0,        0xffffff, 0,       0x808080,
                            0xffffff, 0xffffff, 0xffffff, 0xffffff};
  uint32_t out[9];
  (void)aa_block(1, core, out);
  (void)wedge_table();
  (void)notch_coverage_table();
}
// A compact hooked bowl: a flat upper return curves into a two-cell counter.
// A dash or an F terminal has the same first 3x3, but not this complete 5x5.
// These are authored geometry rules, not inferred letters or learned labels.
std::array<int,2> bowl_point(int u,int v,int orientation) {
  if(orientation&4)u=-u;
  switch(orientation&3) {
  case 1:return {-v,u};case 2:return {-u,-v};case 3:return {v,-u};
  default:return {u,v};
  }
}
bool hooked_bowl(const Frame &f,int x,int y,int orientation) {
  constexpr char shape[5][6]={"####.","...##","#####","#..##","###.#"};
  auto at=[&](int u,int v){const auto p=bowl_point(u,v,orientation);return pixel(f,x+p[0],y+p[1]);};
  const auto fg=at(0,0),bg=at(-2,-1);
  if(distance(fg,bg)<3600)return false;
  // Exact two-surface geometry keeps nearby hues, shading and mixed/AA
  // samples out of this binary rule. All witnesses are actual hinted pixels.
  for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u) {
    const auto p=bowl_point(u,v,orientation);
    if(!f.text[size_t(y+p[1])*f.w+x+p[0]] ||
       at(u,v)!=(shape[v+2][u+2]=='#'?fg:bg))return false;
  }
  // The return alone cannot prove that the upper run ends here. Its tip's
  // own actual 3x3 must show a one-cell terminal with field above/below.
  // This makes the union's source footprint reach radius3 at this anchor.
  for(int v=-1;v<=1;++v)for(int u=-1;u<=1;++u) {
    const auto p=bowl_point(u-2,v-2,orientation);
    const int xx=x+p[0],yy=y+p[1];
    if(xx<0 || yy<0 || xx>=f.w || yy>=f.h || !f.text[size_t(yy)*f.w+xx] ||
       pixel(f,xx,yy)!=(!v && u>=0?fg:bg))return false;
  }
  return true;
}
void bowl_render(const Frame &f,int x,int y,int code,uint32_t *out) {
  const int orientation=(code>>2)&7,role=code&3;
  // The counter's bottom-right contact carries the continuing stem. Keep
  // its source ink solid rather than fading the only connecting subpixel.
  if(role==3){std::fill_n(out,9,pixel(f,x,y));return;}
  auto at=[&](int u,int v){const auto p=bowl_point(u,v,orientation);return pixel(f,x+p[0],y+p[1]);};
  const auto fg=rgb(role?at(0,-1):at(0,0)),bg=rgb(role?at(0,0):at(-1,0));
  // Preserve the source-supported flat run and thin counter. Only the tip
  // corners and the four counter corners receive subpixel coverage.
  constexpr int cap[9]={64,224,256,256,256,256,64,224,256};
  for(int v=-1;v<=1;++v)for(int u=-1;u<=1;++u) {
    const int coverage=role?(std::abs(v)==1 && u==(role==1?-1:1)?96:0):cap[(v+1)*3+u+1];
    const auto p=bowl_point(u,v,orientation);
    out[(p[1]+1)*3+p[0]+1]=blend(bg,fg,coverage);
  }
}
void prepare_ufo(Frame &f) {
  f.ufo_rules.assign(size_t(f.w) * f.h, 0);
  f.aa_planes.clear();
  f.aa_ties.clear();
  if (f.w < 5 || f.h < 5)
    return;
  std::vector<AaTie> mixed_corners;
  std::vector<size_t> square_junctions;
  std::vector<Corner> corners;
  std::vector<std::array<int,3>> axis_candidates,joint_candidates;
  std::vector<std::array<int,3>> bowl_candidates;
  std::vector<std::array<int, 2>> short_join_gaps;
  auto marked = [&](int x, int y) { return f.text[size_t(y) * f.w + x] != 0; };
  for (int y = 2; y < f.h - 2; ++y)
    for (int x = 2; x < f.w - 2; ++x) {
      const bool hinted=marked(x,y) && marked(x-1,y) && marked(x+1,y) &&
                        marked(x,y-1) && marked(x,y+1);
      int coverage_gate=-1;
      auto local_coverage=[&]() {if(coverage_gate<0)coverage_gate=coverage_core(f,x,y);return coverage_gate!=0;};
      if(!hinted) {
        // Do not cross an explicit hint boundary. Flat cores cannot carry AA
        // evidence; the cached key is the inexpensive common rejection.
        const auto key=f.keys[size_t(y)*f.w+x];
        const bool join=junction_key(key);
        if(!key || (!join && !general_candidate(f,x,y) &&
                     !(mixed_corner_key(key) && mixed_corner_candidate(f,x,y)) && !local_coverage()))continue;
        bool clear=true;
        for(int dy=-2;dy<=2 && clear;++dy)for(int dx=-2;dx<=2;++dx)
          if(marked(x+dx,y+dy)){clear=false;break;}
        if(!clear)continue;
      }
      // Keep a sparse set of witnessed square blocks, then reassert it
      // after corner collection. Later AA units cannot round these joins.
      bool join_scope=true;
      for(int dy=-2;hinted && dy<=2 && join_scope;++dy)for(int dx=-2;dx<=2;++dx)
        if(marked(x+dx,y+dy)!=hinted){join_scope=false;break;}
      const auto join_key=f.keys[size_t(y)*f.w+x];
      if(join_scope && junction_key(join_key)) {
        auto protect=[&](int xx,int yy) {
          square_junctions.push_back(size_t(yy)*f.w+xx);
          f.ufo_rules[size_t(yy)*f.w+xx]=5;
        };
        if(general_junction(f,x,y) || local_square_junction(f,x,y) || general_wide_cross(f,x,y)) {
          for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)protect(x+dx,y+dy);
          continue;
        }
        int arms=0;
        for(auto d:std::array<std::array<int,2>,4>{{{{0,1}},{{1,0}},{{0,-1}},{{-1,0}}}})
          arms+=near(pixel(f,x+d[0],y+d[1]),pixel(f,x,y));
        BoldJoin full_join;
        if(bold_join(f,x,y,pixel(f,x,y),arms,full_join)) {
          for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)protect(x+dx,y+dy);
          if(full_join.cap_x>=0)protect(full_join.cap_x,full_join.cap_y);
        }
        JoinCells join;
        if(masked_bold_join(f,x,y,pixel(f,x,y),join_key,join))
          for(unsigned i=0;i<join.count;++i)protect(join.points[i][0],join.points[i][1]);
        std::array<std::array<int,2>,8> cross;
        if(cross_join(f,x,y,pixel(f,x,y),join_key&0xa5u,cross))
          for(auto p:cross)protect(p[0],p[1]);
      }
      AaTie coverage;
      const bool notched=!hinted && join_scope && general_notch_key(join_key) && general_notch(f,x,y,coverage);
      const bool covered=notched || (!hinted && join_scope && local_coverage() && coverage_block(f,x,y,coverage));
      if(covered)mixed_corners.push_back(coverage);
      if(!covered && join_scope && mixed_corner_key(join_key) && mixed_corner_candidate(f,x,y)) {
        AaTie candidate;bool found=false,ambiguous=false;
        for(int kind=1;kind<=8;++kind) {
          AaTie next;if(!mixed_corner(f,x,y,kind,next))continue;
          if(found && candidate.pixels!=next.pixels)ambiguous=true;
          else {candidate=next;found=true;}
        }
        if(found && !ambiguous)mixed_corners.push_back(candidate);
      }
      collect_aa_plane(f,x,y,hinted);
      const auto center=pixel(f,x,y);
      for(int axis:{9,10}) {
        const auto first=axis==9?pixel(f,x,y-1):pixel(f,x-1,y),
                   second=axis==9?pixel(f,x,y+1):pixel(f,x+1,y);
        // Full endpoint groups imply more than 1024 squared separation here.
        if(distance(first,second)<1024)continue;
        Axial p;if(axial(f,x,y,axis==10,p))axis_candidates.push_back({x,y,axis});
      }
      const auto nw=pixel(f,x-1,y-1),ne=pixel(f,x+1,y-1),
                 sw=pixel(f,x-1,y+1),se=pixel(f,x+1,y+1);
      if((nw^ne)|(nw^sw)|(nw^se)) {
        for(int k=0;k<4;++k) {Joint p;if(joint(f,x,y,k,p)) {joint_candidates.push_back({x,y,k});break;}}
      }
      if(!hinted) {
        // Geometric AA families above are independent of glyph meaning.
        // Match the clean convex-cut family separately. Square joins were
        // handled above; S borrowing, counters, R wedges and serifs stay hinted.
        for(int kind=1;kind<=4;++kind) {
          Corner p;
          if(corner(f,x,y,kind,p) && (general_corner(f,p) || general_thin_corner(f,p)))corners.push_back(p);
        }
        continue;
      }
      // A two-cell enclosed counter has two straight parallel bars and
      // foreground at both ends. Its six direct boundary cells prove closure;
      // diagonal cap cells may be clipped or lead into a descender. The
      // counter color must return beyond both one-pixel bars: an isolated
      // two-pixel foreground dot/dash has the opposite field polarity.
      // Preserve these inner gaps with the existing late source-copy case.
      const unsigned counter_key=f.keys[size_t(y)*f.w+x];
      if(counter_key==0x63 || counter_key==0x1d || counter_key==0xc6 || counter_key==0xb8) {
        constexpr unsigned keys[8]={0x63,0x1d,0xc6,0xb8,0xc6,0xb8,0x63,0x1d};
        for(int k=0;k<8;++k)if(counter_key==keys[k] && hooked_bowl(f,x,y,k)) {
          bowl_candidates.push_back({x,y,k});break;
        }
      }
      int ux=0,uy=0;
      if (f.threshold[size_t(y)*f.w+x]>=(3600u>>5)) switch (counter_key) {
      case 206:case 207:case 238:case 239:ux=1;break;
      case 115:case 119:case 243:case 247:ux=-1;break;
      case 186:case 187:case 190:case 191:uy=1;break;
      case 93:case 125:case 221:case 253:uy=-1;break;
      default:break;
      }
      if (ux || uy) {
        const int vx=-uy,vy=ux;
        auto at=[&](int u,int v) {return pixel(f,x+u*ux+v*vx,y+u*uy+v*vy);};
        const auto foreground=at(0,-1);
        bool supported=at(1,0)==center && distance(foreground,center)>=3600;
        if (supported)
          for (int sign : {-1,1})
            for (int u : {0,1}) supported &= at(u,2*sign)==center;
        // The exterior field must also pass beside the near cap. This
        // rejects i/colon dots whose neighboring lines mimic the returns.
        // Either endpoint may anchor the pair, keeping rotations symmetric.
        supported &= at(-2,-1)==center && at(-2,1)==center;
        constexpr int cells[6][2]={{-1,0},{2,0},{0,-1},{1,-1},{0,1},{1,1}};
        if (supported)
          for (auto cell : cells) {
            const int xx=x+cell[0]*ux+cell[1]*vx,
                      yy=y+cell[0]*uy+cell[1]*vy;
            if (!marked(xx,yy) || !near(pixel(f,xx,yy),foreground)) {
              supported=false;break;
            }
          }
        if (supported && marked(x+ux,y+uy)) {
          short_join_gaps.push_back({x,y});
          short_join_gaps.push_back({x+ux,y+uy});
        }
      }

      // Three-cell, one-row counters with observed exterior field returns.
      if (f.threshold[size_t(y)*f.w+x]>=(3600u>>5) &&
          (counter_key==0xe7u || counter_key==0xbdu)) {
        const int ux=counter_key==0xe7u?1:0,uy=1-ux,vx=-uy,vy=ux;
        auto at=[&](int u,int v){return pixel(f,x+u*ux+v*vx,y+u*uy+v*vy);};
        const auto foreground=at(0,-1);
        bool supported=distance(center,foreground)>=3600;
        for (int u=-1;u<=1;++u) {
          supported &= at(u,0)==center && at(u,-2)==center && at(u,2)==center;
          for (int v : {-1,1})
            supported &= near(at(u,v),foreground) && marked(x+u*ux+v*vx,y+u*uy+v*vy);
        }
        for (int u : {-2,2})
          supported &= near(at(u,0),foreground) && marked(x+u*ux,y+u*uy);
        supported &= (at(-2,-1)==center && at(2,-1)==center) ||
                     (at(-2,1)==center && at(2,1)==center);
        if (supported)
          for (int u=-1;u<=1;++u)
            if (marked(x+u*ux,y+u*uy)) short_join_gaps.push_back({x+u*ux,y+u*uy});
      }
      // Flat terminals of actual one-pixel bars, at least three pixels long.
      if (f.threshold[size_t(y)*f.w+x]>=(3600u>>5)) {
        int vx=0,vy=0;
        switch(counter_key) {
        case 239:vx=1;break;case 247:vx=-1;break;
        case 191:vy=1;break;case 253:vy=-1;break;
        default:break;
        }
        if (vx || vy) {
          const int ux=-vy,uy=vx;
          auto at=[&](int u,int v){return pixel(f,x+u*ux+v*vx,y+u*uy+v*vy);};
          const auto background=at(0,-1);
          bool supported=distance(center,background)>=3600 && near(at(0,1),center) &&
                         near(at(0,2),center) && near(at(0,-2),background);
          for (int v=-1;v<=2;++v)
            for (int u : {-1,1}) supported &= near(at(u,v),background);
          if (supported) short_join_gaps.push_back({x,y});
          if (!supported) {
            // A one-cell square end of a bar attached to a two-cell stem.
            // Background texture is irrelevant; every excluded cell must
            // remain well separated from the repeated foreground stem.
            bool cap=true;
            for(int v=1;v<=2;++v)for(int u=-2;u<=2;++u)
              cap &= near(at(u,v),center);
            for(int v=-2;v<=0;++v)for(int u=-2;u<=2;++u)
              if(u || v) cap &= distance(at(u,v),center)>=3600;
            if(cap) {
              short_join_gaps.push_back({x,y});
              for(int sign:{-1,1})
                if(marked(x+sign*ux,y+sign*uy))
                  short_join_gaps.push_back({x+sign*ux,y+sign*uy});
            }
          }
        }
      }
      // Short square serif joins, including a diagonal departure after the
      // first stem pixel. A clipped curve join only copies its middle block.
      if (f.threshold[size_t(y)*f.w+x]>=(3600u>>5)) {
        int vx=0,vy=0;
        switch(counter_key) {
        case 167:vy=1;break;case 229:vy=-1;break;
        case 173:vx=1;break;case 181:vx=-1;break;
        default:break;
        }
        if (vx || vy) {
          const int ux=-vy,uy=vx;
          auto at=[&](int u,int v){return pixel(f,x+u*ux+v*vx,y+u*uy+v*vy);};
          const auto background=at(0,-1);
          bool common=distance(center,background)>=3600;
          for (int u=-1;u<=1;++u) {
            common &= near(at(u,0),center) && near(at(u,-1),background);
            common &= near(at(u,1),u?background:center);
          }
          common &= near(at(-2,0),background) && near(at(2,0),background);
          int tail=0;bool tail_valid=true;
          for (int u=-1;u<=1;++u) {
            if (near(at(u,2),center)) ++tail;
            else if (!near(at(u,2),background)) tail_valid=false;
          }
          if (common && tail_valid && tail==1) {
            bool serif=true,curve=true;
            for (int sign : {-1,1}) {
              serif &= near(at(2*sign,-1),background) && near(at(2*sign,1),background);
              curve &= near(at(2*sign,-1),background) && near(at(2*sign,1),center) &&
                       near(at(2*sign,2),center);
            }
            if (serif)
              for (int v=0;v<=1;++v)
                for (int u=-1;u<=1;++u)
                  serif &= marked(x+u*ux+v*vx,y+u*uy+v*vy);
            if (serif)
              for (int v=0;v<=1;++v)
                for (int u=-1;u<=1;++u) {
                  const int xx=x+u*ux+v*vx,yy=y+u*uy+v*vy;
                  if (marked(xx,yy)) short_join_gaps.push_back({xx,yy});
                }
            else if (curve) short_join_gaps.push_back({x,y});
          }
        }
      }

      JoinCells masked_join;
      const bool masked=f.threshold[size_t(y)*f.w+x]>=(3600u>>5) &&
          masked_bold_join(f,x,y,center,f.keys[size_t(y)*f.w+x],masked_join);
      if (!masked) {
        // Square T/F junctions require a coherent diagonal background and exactly
        // three connected axial arms, with no evidence of a diagonal
        // continuation.
        Color sum{};
        for (int yy : {-1, 1})
          for (int xx : {-1, 1}) {
            auto c = rgb(pixel(f, x + xx, y + yy));
            sum.r += c.r;
            sum.g += c.g;
            sum.b += c.b;
          }
        uint32_t bg = packed({(sum.r + 2) / 4, (sum.g + 2) / 4, (sum.b + 2) / 4});
        bool clear = true, classified = true, diagonal = false;
        int arms = 0;
        for (int yy : {-1, 1})
          for (int xx : {-1, 1})
            clear &= near(pixel(f, x + xx, y + yy), bg);
        for (auto d : std::array<std::array<int, 2>, 4>{
                 {{{0, 1}}, {{1, 0}}, {{0, -1}}, {{-1, 0}}}}) {
          int dx = d[0], dy = d[1];
          auto c = pixel(f, x + dx, y + dy);
          bool on = near(c, center), off = near(c, bg);
          arms += on;
          classified &= on || off;
          bool straight = near(pixel(f, x + 2 * dx, y + 2 * dy), center);
          for (int sign : {-1, 1})
            diagonal |=
                on && !straight &&
                near(pixel(f, x + 2 * dx - sign * dy, y + 2 * dy + sign * dx),
                     center);
        }
        const int contrast = distance(center, bg);
        const bool exact_join = clear && classified && !diagonal && arms == 3 &&
                                contrast >= 1800;
        if (exact_join)
          for (int yy = -1; yy <= 1; ++yy)
            for (int xx = -1; xx <= 1; ++xx)
              if (marked(x + xx, y + yy))
                f.ufo_rules[size_t(y + yy) * f.w + x + xx] = 5;
        // A thin junction can meet several other surfaces. Its center layer
        // is still supported when all three arms continue two pixels and
        // every diagonal/missing arm is well separated. This also preserves
        // a narrow background gap between glyphs above a textured field.
        // Copy only this junction, leaving all neighboring blocks untouched.
        if (!exact_join && arms == 3 &&
            (f.keys[size_t(y) * f.w + x] & 0xa5u) == 0xa5u &&
            f.threshold[size_t(y) * f.w + x] >= (1800u >> 5)) {
          bool supported = true;
          for (int yy : {-1, 1})
            for (int xx : {-1, 1})
              supported &= distance(pixel(f, x + xx, y + yy), center) >= 1800;
          for (auto d : std::array<std::array<int, 2>, 4>{
                   {{{0, 1}}, {{1, 0}}, {{0, -1}}, {{-1, 0}}}}) {
            auto c = pixel(f, x + d[0], y + d[1]);
            supported &= near(c, center)
                             ? near(pixel(f, x + 2*d[0], y + 2*d[1]), center)
                             : distance(c, center) >= 1800;
          }
          if (supported)
            f.ufo_rules[size_t(y) * f.w + x] = 5;
        }
        // A short side bar need not continue a second pixel. A five-pixel
        // straight stem and two continued fields beside that bar instead
        // support square inner gaps. Save only those two field blocks; the
        // established AA/notch/field-gap interpretations retain priority.
        if (!exact_join && arms == 3 &&
            (f.keys[size_t(y)*f.w+x] & 0xa5u) == 0xa5u &&
            f.threshold[size_t(y)*f.w+x] >= (1800u >> 5)) {
          // Exactly three near arms identify one opposing stem pair, so
          // derive its orientation once instead of testing four rotations.
          const bool horizontal=near(pixel(f,x-1,y),center) &&
                                near(pixel(f,x+1,y),center);
          const int bx=horizontal ? 0 : (near(pixel(f,x+1,y),center) ? 1 : -1),
                    by=horizontal ? (near(pixel(f,x,y+1),center) ? 1 : -1) : 0,
                    tx=-by,ty=bx;
          bool supported=distance(pixel(f,x-bx,y-by),center)>=1800 &&
              near(pixel(f,x+2*tx,y+2*ty),center) &&
              near(pixel(f,x-2*tx,y-2*ty),center);
          if (supported) {
            for (int sign : {-1,1}) {
              const auto field=pixel(f,x+bx+sign*tx,y+by+sign*ty);
              supported &= distance(field,center)>=1800 &&
                  distance(pixel(f,x-bx+sign*tx,y-by+sign*ty),center)>=1800 &&
                  near(pixel(f,x+2*bx+sign*tx,y+2*by+sign*ty),field);
            }
            if (supported)
              for (int sign : {-1,1})
                short_join_gaps.push_back({x+bx+sign*tx,y+by+sign*ty});
          }
        }
        // A valid join has a diagonal background sample at distance >=1800.
        // Reuse the cached key/threshold to reject weaker patches before the
        // rare continuation checks (threshold = max(48, max_core_distance >> 5)).
        BoldJoin join;
        if (!exact_join && arms >= 3 &&
            (f.keys[size_t(y) * f.w + x] & 0xa5u) &&
            f.threshold[size_t(y) * f.w + x] >= (1800u >> 5)) {
          std::array<std::array<int, 2>, 8> cross;
          if (arms == 4 && cross_join(f, x, y, center,
                                      f.keys[size_t(y) * f.w + x] & 0xa5u, cross)) {
            for (const auto &p : cross)
              if (marked(p[0], p[1]))
                f.ufo_rules[size_t(p[1]) * f.w + p[0]] = 5;
          } else if (bold_join(f, x, y, center, arms, join)) {
            if (marked(join.x0, join.y0))
              f.ufo_rules[size_t(join.y0) * f.w + join.x0] = 5;
            if (marked(join.x1, join.y1))
              f.ufo_rules[size_t(join.y1) * f.w + join.x1] = 5;
            if (join.cap_x >= 0 && marked(join.cap_x, join.cap_y))
              f.ufo_rules[size_t(join.cap_y) * f.w + join.cap_x] = 5;
          }
        }
        // Shaded bitmap fonts vary along an otherwise straight stroke. Keep
        // the same junction geometry, but measure endpoint tolerance against
        // its local foreground/background contrast. Claim only foreground
        // arms here: copying a textured background would erase real detail.
        if (!exact_join && contrast >= 3600) {
          const int tolerance = std::max(144, contrast / 16);
          auto close = [&](uint32_t a, uint32_t b) {
            return distance(a, b) <= tolerance;
          };
          bool classified = true, shaded = false;
          unsigned arm_mask = 0;
          constexpr int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0};
          for (int d = 0; d < 4; ++d) {
            auto c = pixel(f, x + dx[d], y + dy[d]);
            int delta = distance(c, center);
            bool on = delta <= tolerance, off = close(c, bg);
            shaded |= on && delta > 144;
            if (on)
              arm_mask |= 1u << d;
            classified &= on || off;
          }
          if (shaded && classified && __builtin_popcount(arm_mask) == 3) {
            bool supported = true;
            for (int yy : {-1, 1})
              for (int xx : {-1, 1})
                supported &= close(pixel(f, x + xx, y + yy), bg);
            for (int d = 0; supported && d < 4; ++d)
              if ((arm_mask & (1u << d)) &&
                  distance(pixel(f, x + 2 * dx[d], y + 2 * dy[d]), center) >
                      contrast / 9)
                for (int sign : {-1, 1})
                  supported &= !close(pixel(f, x + 2 * dx[d] - sign * dy[d],
                                            y + 2 * dy[d] + sign * dx[d]), center);
            if (supported) {
              f.ufo_rules[size_t(y) * f.w + x] = 5;
              for (int d = 0; d < 4; ++d)
                if ((arm_mask & (1u << d)) && marked(x + dx[d], y + dy[d]))
                  f.ufo_rules[size_t(y + dy[d]) * f.w + x + dx[d]] = 5;
            }
          }
        }
      } else {
        // A mixed-width core cannot be an old thin/exact or full-cross case.
        // Keep the old full-bold case's extra witnessed cap when applicable.
        BoldJoin legacy;
        if (bold_join(f,x,y,center,masked_join.arms,legacy)) {
          for (auto p : std::array<std::array<int,2>,3>{{
                 {{legacy.x0,legacy.y0}},{{legacy.x1,legacy.y1}},
                 {{legacy.cap_x,legacy.cap_y}}}})
            if (p[0]>=0 && marked(p[0],p[1]))
              f.ufo_rules[size_t(p[1])*f.w+p[0]]=5;
        }
        for (unsigned i=0;i<masked_join.count;++i) {
          auto p=masked_join.points[i];
          if (marked(p[0],p[1])) f.ufo_rules[size_t(p[1])*f.w+p[0]]=5;
        }
      }
      for (int kind = 1; kind <= 4; ++kind) {
        Corner p;
        if (corner(f, x, y, kind, p))
          corners.push_back(p);
      }
    }
  // A clipped S corner can lack one direct continuation. Admit it only when
  // two radius-two solid corners support the same two surface colors.
  std::vector<Corner> accepted;
  // A new endpoint must not steal a previously supported interpretation or
  // become a distant witness that changes admission of old clipped corners.
  std::stable_sort(corners.begin(), corners.end(),
                   [](auto a, auto b) {
                     if (a.extended_endpoint != b.extended_endpoint)
                       return !a.extended_endpoint;
                     return a.solid > b.solid;
                   });
  std::set<size_t> used;
  std::unordered_map<size_t, std::vector<const Corner *>> solid;
  for (const auto &p : corners)
    if (marked(p.x,p.y) && p.solid && !p.extended_endpoint)
      solid[size_t(p.y) * f.w + p.x].push_back(&p);
  for (auto p : corners) {
    bool h = false, v = false;
    auto supports = [&](int x, int y) {
      auto it = solid.find(size_t(y) * f.w + x);
      if (it == solid.end())
        return false;
      for (auto q : it->second)
        if ((near(p.fg, q->fg) && near(p.bg, q->bg)) ||
            (near(p.fg, q->bg) && near(p.bg, q->fg)))
          return true;
      return false;
    };
    if (!p.solid) {
      h = supports(p.x + 2 * p.dx, p.y);
      v = supports(p.x, p.y + 2 * p.dy);
    }
    // Two clipped inner corners can support each other in a short S bend.
    // Require opposite orientations, consistent mixture colors, and a solid
    // outer corner at each end. The old direct-solid test alone depended on
    // unrelated letters two pixels to the left/right of the glyph.
    auto clipped_pair = [&] {
      auto anchor = [&](int x, int y, int dx, int dy) {
        if (x < 2 || y < 2 || x >= f.w - 2 || y >= f.h - 2)
          return false;
        auto it = solid.find(size_t(y) * f.w + x);
        if (it == solid.end())
          return false;
        for (auto q : it->second)
          if (q->dx == dx && q->dy == dy && near(p.fg, q->fg) &&
              near(p.bg, q->bg))
            return true;
        return false;
      };
      for (bool vertical : {false, true}) {
        int x = p.x + (vertical ? 0 : 2 * p.dx),
            y = p.y + (vertical ? 2 * p.dy : 0);
        if (x < 2 || y < 2 || x >= f.w - 2 || y >= f.h - 2 ||
            !marked(x, y) || !marked(x - 1, y) || !marked(x + 1, y) ||
            !marked(x, y - 1) || !marked(x, y + 1))
          continue;
        Corner q;
        if (!corner(f, x, y, 5 - p.kind, q) || q.solid ||
            !near(p.fg, q.fg) || !near(p.bg, q.bg) ||
            !near(pixel(f, p.x, p.y), pixel(f, q.x, q.y)))
          continue;
        if (vertical) {
          if (anchor(p.x + 2 * p.dx, p.y, -p.dx, p.dy) &&
              anchor(q.x + 2 * q.dx, q.y, -q.dx, q.dy))
            return true;
        } else if (anchor(p.x, p.y + 2 * p.dy, p.dx, -p.dy) &&
                   anchor(q.x, q.y + 2 * q.dy, q.dx, -q.dy))
          return true;
      }
      return false;
    };
    if ((p.solid || (h && v) || clipped_pair()) &&
        used.insert(size_t(p.y) * f.w + p.x).second)
      accepted.push_back(p);
  }
  for (auto p : accepted) {
    // The new weak/near-solid ratios locate a subpixel cut inside this block;
    // they do not justify replacing adjoining contour coverage with a copy.
    if (p.extended_endpoint)
      continue;
    for (auto d : std::array<std::array<int, 2>, 4>{
             {{{0, 1}}, {{1, 0}}, {{0, -1}}, {{-1, 0}}}}) {
      int xx = p.x + d[0], yy = p.y + d[1];
      if ((marked(p.x,p.y)?marked(xx,yy):!marked(xx,yy)) && straight_corner_neighbor(f,p,d[0],d[1]))
        f.ufo_rules[size_t(yy) * f.w + xx] = 5;
    }
  }
  for (auto p : accepted)
    f.ufo_rules[size_t(p.y) * f.w + p.x] =
        p.field_gap ? uint8_t(5 | FIELD_GAP_PRIORITY) : uint8_t(p.kind);
  // A source-supported R notch uses a symmetric wedge, not a rounded knob.
  std::vector<std::array<int,4>> rotated_notches;
  for (int y = 2; y < f.h - 2; ++y)
    for (int x = 2; x < f.w - 2; ++x) {
      if (!marked(x,y)) continue;
      bool legacy = false;
      if (marked(x,y-1) && marked(x,y+1)) {
        Wedge p;
        if (wedge(f, x, y, p)) {
          legacy = true;
          // A whole notch stencil must not repaint a confirmed field gap.
          // Veto it as a unit rather than clipping one side asymmetrically.
          const auto flags = f.ufo_rules[size_t(y - 1) * f.w + x] |
                             f.ufo_rules[size_t(y) * f.w + x] |
                             f.ufo_rules[size_t(y + 1) * f.w + x];
          if (flags & FIELD_GAP_PRIORITY)
            continue;
          for (int yy = -1; yy <= 1; ++yy)
            f.ufo_rules[size_t(y + yy) * f.w + x] = uint8_t(7 + yy);
        }
      }
      if (legacy) continue;
      // Only these cached 3x3 keys can contain a thin near-background fork.
      // Keep the old right-facing matcher first, including its gap veto.
      int rotation = 0;
      Wedge p;
      switch (f.keys[size_t(y)*f.w+x]) {
      case 0x1a:
        if (marked(x-1,y) && marked(x+1,y) && wedge<1>(f,x,y,p)) rotation=1;
        break;
      case 0x52:
        if (marked(x,y-1) && marked(x,y+1) && wedge<2>(f,x,y,p)) rotation=2;
        break;
      case 0x58:
        if (marked(x-1,y) && marked(x+1,y) && wedge<3>(f,x,y,p)) rotation=3;
        break;
      }
      if (!rotation) continue;
      rotated_notches.push_back({x,y,rotation,0});
    }
  // An old notch visited later in raster order must veto the entire new
  // stencil. Apply only these rare deferred claims, without another frame
  // scan or a new persistent per-pixel buffer.
  constexpr int tx[]={0,-1,0,1},ty[]={1,0,-1,0};
  size_t eligible=0;
  for (auto &claim : rotated_notches) {
      const int x=claim[0],y=claim[1],rotation=claim[2];
      bool clear=true;
      for (int i=-1;i<=1;++i) {
        const auto rule=f.ufo_rules[size_t(y+i*ty[rotation])*f.w+x+i*tx[rotation]];
        clear &= !(rule & FIELD_GAP_PRIORITY) && (rule & 15u)<6;
      }
      if (clear) {claim[3]=1;++eligible;}
  }
  if (eligible>1) {
    // Mark all eligible anchors before deciding any overlap. Removing an
    // ambiguous proposal greedily would let a one-pixel edit rephase a
    // whole chain of otherwise unchanged strokes. Bit6 is preparation-only.
    constexpr uint8_t CANDIDATE=0x40;
    for (const auto &claim : rotated_notches)
      if (claim[3]) f.ufo_rules[size_t(claim[1])*f.w+claim[0]]|=CANDIDATE;
    for (auto &claim : rotated_notches) {
      if (!claim[3]) continue;
      const int x=claim[0],y=claim[1];
      bool conflict=false;
      // Two three-block strips can only meet within this radius. This is
      // bounded local work per rare claim, not a quadratic list comparison
      // or a connected-component decision.
      for (int dy=-2;dy<=2 && !conflict;++dy)
        for (int dx=-2;dx<=2;++dx) {
          if ((!dx && !dy) ||
              !(f.ufo_rules[size_t(y+dy)*f.w+x+dx]&CANDIDATE)) continue;
          const auto key=f.keys[size_t(y+dy)*f.w+x+dx];
          const int other=key==0x1a?1:key==0x52?2:3;
          const bool horizontal=claim[2]!=2,other_horizontal=other!=2;
          const bool overlaps=horizontal==other_horizontal
              ? (horizontal?dy==0:dx==0)
              : (std::abs(dx)<=1 && std::abs(dy)<=1);
          if (overlaps) {conflict=true;break;}
        }
      if (conflict) claim[3]=2;
    }
    for (const auto &claim : rotated_notches)
      if (claim[3]) f.ufo_rules[size_t(claim[1])*f.w+claim[0]]&=uint8_t(~CANDIDATE);
  }
  for (const auto &claim : rotated_notches) {
    if (claim[3]!=1) continue;
    const int x=claim[0],y=claim[1],rotation=claim[2];
    for (int i=-1;i<=1;++i)
      f.ufo_rules[size_t(y+i*ty[rotation])*f.w+x+i*tx[rotation]]=
          uint8_t((rotation<<4)|(7+i));
  }
  for (auto p : short_join_gaps) {
    auto &rule=f.ufo_rules[size_t(p[1])*f.w+p[0]];
    if (marked(p[0],p[1]) && !rule) rule=5;
  }
  std::sort(square_junctions.begin(),square_junctions.end());
  square_junctions.erase(std::unique(square_junctions.begin(),square_junctions.end()),square_junctions.end());
  for(auto i:square_junctions)f.ufo_rules[i]=5;
  // Geometry proposals include their opaque source fields. A shared edge
  // renders from its own samples, so compatible overlaps need no averaging.
  struct Claim {int x,y,k,family;uint8_t previous;Joint p;};
  std::vector<Claim> proposals;
  auto role_at=[&](const Claim &c,int dx,int dy) {
    if(std::abs(dx)>2 || std::abs(dy)>2)return 0;
    return c.family==9?(dy==0?9:5):c.family==10?(dx==0?10:5):joint_role(c.p,c.k,dx,dy);
  };
  auto render_class=[&](const Claim &c,int role) {
    if(role==5 || role==9 || role==10)return role;
    if(role>=12)return role<=13?9:10;
    return c.p.a==0?10:c.p.b==0?9:11;
  };
  auto render_claim=[&](const Claim &c,int x,int y,int role,uint32_t *out) {
    const int family=render_class(c,role);
    if(family==5)std::fill_n(out,9,pixel(f,x,y));
    else if(family==9 || family==10)local_edge(f,x,y,family==10?1:0,family==9?1:0,out);
    else joint_render(c.p,c.k,11,out);
  };
  auto propose=[&](int x,int y,int k,int family,Joint p) {
    Claim c{x,y,k,family,f.ufo_rules[size_t(y)*f.w+x],p};bool clear=true;
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
      const auto r=f.ufo_rules[size_t(y+dy)*f.w+x+dx];const int role=role_at(c,dx,dy);
      const bool scope=marked(x,y)?marked(x+dx,y+dy):!marked(x+dx,y+dy);
      clear &= scope && ((r&15)<5 || (r&15)>=9 || (role==5 && (r&15)==5));
    }
    if(clear)proposals.push_back(c);
  };
  for(const auto &q:axis_candidates)propose(q[0],q[1],0,q[2],{});
  for(const auto &q:joint_candidates) {Joint p;if(joint(f,q[0],q[1],q[2],p))propose(q[0],q[1],q[2],11,p);}
  auto index=[&](const Claim &p){return size_t(p.y)*f.w+p.x;};
  std::sort(proposals.begin(),proposals.end(),[&](const Claim&a,const Claim&b){return index(a)<index(b);});
  // Competing geometry types at the same anchor are ambiguous, not a
  // reason to give the first raster encounter priority.
  for(size_t i=1;i<proposals.size();++i)if(index(proposals[i])==index(proposals[i-1])) {
    proposals[i].family=0;proposals[i-1].family=0;
  }
  proposals.erase(std::remove_if(proposals.begin(),proposals.end(),[](const Claim &c){return !c.family;}),proposals.end());
  for(const auto &c:proposals)f.ufo_rules[index(c)]=uint8_t(64|(c.k<<4)|c.family);
  for(const auto &c:proposals) {
    // Full 5x5 stencils can overlap only when their anchors are within4.
    // Compare each pair once. Most dense axial runs share one straight
    // edge and are compatible without rendering any candidate pixels.
    for(int oy=-4;oy<=4;++oy)for(int ox=-4;ox<=4;++ox) {
      const int ax=c.x+ox,ay=c.y+oy;
      if(ax<2 || ay<2 || ax>=f.w-2 || ay>=f.h-2)continue;
      const size_t other_index=size_t(ay)*f.w+ax;if(other_index<=index(c))continue;
      const auto marker=f.ufo_rules[other_index];if(!(marker&64))continue;
      const int other_family=marker&15;
      if(c.family==other_family && ((c.family==9 && !oy) || (c.family==10 && !ox)))continue;
      const auto it=std::lower_bound(proposals.begin(),proposals.end(),other_index,[&](const Claim &a,size_t i){return index(a)<i;});
      if(it==proposals.end() || index(*it)!=other_index)continue;
      const auto &other=*it;bool conflict=false;
      auto compare=[&](int x,int y) {
        const int a=role_at(c,x-c.x,y-c.y),b=role_at(other,x-other.x,y-other.y);if(!a || !b)return;
        const int ac=render_class(c,a),bc=render_class(other,b);
        if(ac==bc && ac!=11)return;
        uint32_t aa[9],bb[9];render_claim(c,x,y,a,aa);render_claim(other,x,y,b,bb);
        conflict |= !std::equal(aa,aa+9,bb);
      };
      // Opaque field overlaps always copy the same source pixel. Only the
      // at most 10 non-copy points across this pair require comparison.
      for(const auto *q:{&c,&other})for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx)
        if(role_at(*q,dx,dy)!=5)compare(q->x+dx,q->y+dy);
      if(conflict) {f.ufo_rules[index(c)]|=128;f.ufo_rules[other_index]|=128;}
    }
  }
  for(auto &c:proposals) {c.k|=f.ufo_rules[index(c)]&128?4:0;f.ufo_rules[index(c)]=c.previous;}
  for(const auto &c:proposals)if(!(c.k&4)) {
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
      const int role=role_at(c,dx,dy);auto &old=f.ufo_rules[size_t(c.y+dy)*f.w+c.x+dx];
      // Compatible source-copy claims win deterministically over a more
      // expensive stencil whose rendered result is identical.
      if(role==5) {if((old&15)!=5)old=5;}
      else if((old&15)!=5)old=uint8_t((c.k<<4)|role);
    }
  }
  // Only accepted axial units may request an exterior end. Existing
  // hardcoded geometry retains priority; the cap writes one proven source
  // copy, and never propagates into another prepared stencil.
  for(const auto &c:proposals)if(!(c.k&4) && (c.family==9 || c.family==10)) {
    const int nx=c.family==10?1:0,ny=c.family==9?1:0,tx=ny,ty=nx;
    for(int sign:{-1,1})for(int side:{-1,1}) {
      const int x=c.x+3*sign*tx,y=c.y+3*sign*ty;
      if(x<2 || y<2 || x>=f.w-2 || y>=f.h-2)continue;
      auto &rule=f.ufo_rules[size_t(y)*f.w+x];if(rule)continue;
      if(opaque_edge_cap(f,x,y,side*nx,side*ny,-sign*tx,-sign*ty))rule=5;
    }
  }
  // Apply whole, compatible bowls after existing AA, gaps and notches. Use
  // sparse sorted writes, not another frame scan or a persistent frame map.
  // Bit6 is free again after the preparation-only conflict markers above.
  struct BowlWrite {size_t index,unit;uint8_t code;};
  std::vector<BowlWrite> bowl_writes;
  std::vector<uint8_t> bowl_ok(bowl_candidates.size(),1);
  constexpr int targets[4][2]={{-2,-2},{-1,1},{0,1},{1,1}};
  for(size_t i=0;i<bowl_candidates.size();++i) {
    const auto &c=bowl_candidates[i];
    for(int role=0;role<4;++role) {
      const auto p=bowl_point(targets[role][0],targets[role][1],c[2]);
      const size_t index=size_t(c[1]+p[1])*f.w+c[0]+p[0];
      const auto rule=f.ufo_rules[index];
      if((rule && rule!=5) || std::binary_search(square_junctions.begin(),square_junctions.end(),index))bowl_ok[i]=0;
      bowl_writes.push_back({index,i,uint8_t(64|(c[2]<<2)|role)});
    }
  }
  std::sort(bowl_writes.begin(),bowl_writes.end(),[](const BowlWrite&a,const BowlWrite&b){return a.index<b.index;});
  // Veto all conflicting units, rather than allowing raster order to choose
  // one. Matching writes are harmless; different orientations may still
  // express identical counter coverage, so compare their actual recipes.
  for(size_t begin=0;begin<bowl_writes.size();) {
    size_t end=begin+1;while(end<bowl_writes.size() && bowl_writes[end].index==bowl_writes[begin].index)++end;
    bool conflict=false;uint32_t first[9];
    const int x=int(bowl_writes[begin].index%f.w),y=int(bowl_writes[begin].index/f.w);
    if(end-begin>1) {
      bowl_render(f,x,y,bowl_writes[begin].code,first);
      for(size_t j=begin+1;j<end;++j) {uint32_t next[9];bowl_render(f,x,y,bowl_writes[j].code,next);conflict|=!std::equal(first,first+9,next);}
    }
    if(conflict)for(size_t j=begin;j<end;++j)bowl_ok[bowl_writes[j].unit]=0;
    begin=end;
  }
  for(const auto &write:bowl_writes)if(bowl_ok[write.unit])f.ufo_rules[write.index]=write.code;
  for(auto i:square_junctions)f.ufo_rules[i]=5;
  finish_aa_planes(f,square_junctions);
  // These fits may refine old single-corner endpoint estimates, but cannot
  // steal a square join, plane/union, gap/notch or bowl unit. A fully covered
  // source elbow stays square; fractional centres use the prepared cache.
  for(const auto &c:mixed_corners) {
    auto &rule=f.ufo_rules[c.index];if(rule && !(rule>=1 && rule<=4))continue;
    const auto own=pixel(f,int(c.index%f.w),int(c.index/f.w));
    if(std::all_of(c.pixels.begin(),c.pixels.end(),[&](uint32_t v){return v==own;}))rule=5;
    else {rule=169;f.aa_ties.push_back(c);}
  }
  std::sort(f.aa_ties.begin(),f.aa_ties.end(),[](const AaTie&a,const AaTie&b){return a.index<b.index;});
}
static int ufo_block_impl(const Frame &f, int x, int y, const uint32_t *core,
              uint32_t *out) {
  int kind = f.ufo_rules[size_t(y) * f.w + x];
  if (!kind)
    return 0;
  if(kind>=144)return aa_plane_block(f,x,y,uint8_t(kind),core[4],out);
  if(kind&64) {bowl_render(f,x,y,kind,out);return (kind&3)==3?2:1;}
  const int rotation=(kind>>4)&3;
  kind &= 15;
  if(kind>=11) {
    const int dx=rotation&1?-1:1,dy=rotation&2?-1:1;
    if(kind>=12) {
      local_edge(f,x,y,kind<=13?0:dx,kind<=13?dy:0,out);
      return std::all_of(out,out+9,[&](uint32_t c){return c==core[4];})?2:1;
    }
    Joint p;if(!joint(f,x,y,rotation,p,false))return 0;
    claim_render(f,x,y,p,rotation,kind,out);
    return std::all_of(out,out+9,[&](uint32_t c){return c==core[4];})?2:1;
  }
  if(kind==9 || kind==10) {
    // Preparation proved the actual 5x5 context; its copied source is
    // immutable during rendering, so no second 25-sample validation is needed.
    local_edge(f,x,y,kind==10?1:0,kind==9?1:0,out);
    return std::all_of(out,out+9,[&](uint32_t c){return c==core[4];})?2:1;
  }
  if (kind <= 4) {
    const int dy=kind<=2?1:-1, dx=(kind==1 || kind==3)?1:-1;
    const auto index=[](int xx,int yy){return (yy+1)*3+xx+1;};
    if (distance(core[index(dx,0)],core[index(0,dy)])>144 ||
        distance(core[index(-dx,0)],core[index(0,-dy)])>144) {
      if (x<2 || y<2 || x>=f.w-2 || y>=f.h-2) return 0;
      uint32_t canonical[9];std::copy_n(core,9,canonical);
      const auto fg=packed(continued_endpoint(f,x,y,dx,dy));
      const auto bg=packed(continued_endpoint(f,x,y,-dx,-dy));
      canonical[index(dx,0)]=canonical[index(0,dy)]=fg;
      canonical[index(-dx,0)]=canonical[index(0,-dy)]=bg;
      return aa_block(kind,canonical,out,true)?1:0;
    }
    return aa_block(kind,core,out,true)?1:0;
  }
  std::fill_n(out, 9, core[4]);
  if (kind == 5)
    return 2;
  constexpr int tx[]={0,-1,0,1},ty[]={1,0,-1,0};
  const int anchor_x=x+(7-kind)*tx[rotation],
            anchor_y=y+(7-kind)*ty[rotation];
  Wedge p;
  bool matched=false;
  switch (rotation) {
  case 0: matched=wedge(f,anchor_x,anchor_y,p); break;
  case 1: matched=wedge<1>(f,anchor_x,anchor_y,p); break;
  case 2: matched=wedge<2>(f,anchor_x,anchor_y,p); break;
  case 3: matched=wedge<3>(f,anchor_x,anchor_y,p); break;
  }
  if (!matched) return 0;
  const auto &cov = wedge_table()[p.alpha];
  // The established coverage stencil is unchanged. Newly admitted ramps use
  // this source row's foreground/shadow instead of flattening three rows to
  // the middle shade. Uniform-color wedges keep their prior exact output.
  if (p.shaded && y != anchor_y) {
    p.fg = rgb(pixel(f,x,y));
    p.bg = rgb(pixel(f,x-1,y));
  }
  for (int oy = 0; oy < 3; ++oy) {
    int row = (kind - 7) * 3 + oy + 2;
    if (row < 0 || row >= 7)
      continue;
    for (int ox = 0; ox < 3; ++ox) {
      int px=ox,py=oy;
      if (rotation==1) {px=2-oy;py=ox;}
      if (rotation==2) {px=2-ox;py=2-oy;}
      if (rotation==3) {px=oy;py=2-ox;}
      out[py * 3 + px] = blend(p.bg, p.fg, cov[row * 3 + ox]);
    }
  }
  return 3;
}
int ufo_block(const Frame &f,int x,int y,const uint32_t *core,uint32_t *out) {
  const int result=ufo_block_impl(f,x,y,core,out);
  if(result && result!=2)bound_source_range(core,out);
  return result;
}
} // namespace lut
