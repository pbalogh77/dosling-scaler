#include "model.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifndef DL_GENERAL_AA_SUPPORT
#define DL_GENERAL_AA_SUPPORT 5
#endif
namespace {
constexpr int SUPPORT_RADIUS=DL_GENERAL_AA_SUPPORT/2;
constexpr int N=13,C=N/2;using Image=std::vector<uint8_t>;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void put(Image&s,int x,int y,lut::Color c){auto i=(y*N+x)*3;s[i]=c.r;s[i+1]=c.g;s[i+2]=c.b;}
Image corner(){Image s(N*N*3);for(int y=0;y<N;++y)for(int x=0;x<N;++x)put(s,x,y,x<=C&&y<=C?lut::Color{240,240,240}:lut::Color{16,16,16});put(s,C,C,{128,128,128});return s;}
Image rotate(const Image&s){Image o=s;for(int y=0;y<N;++y)for(int x=0;x<N;++x)for(int c=0;c<3;++c)o[(x*N+N-1-y)*3+c]=s[(y*N+x)*3+c];return o;}
int block(const Image&s,uint32_t*out,bool hinted=false,lut::Frame* saved=nullptr){lut::Frame f;dl_rect r{0,0,N,N};f.prepare(s.data(),N,N,N*3,hinted?&r:nullptr,hinted?1:0,true);lut::prepare_ufo(f);uint32_t core[9];f.core(C,C,core);int k=lut::ufo_block(f,C,C,core,out);check(std::count(f.text.begin(),f.text.end(),uint8_t(1))==(hinted?N*N:0),"AA recognition mutated the caller hint mask");if(saved)*saved=std::move(f);return k;}
double cdf(double z){auto sq=[](double a){return a>0?a*a:0.;};return std::clamp((sq(z)-2*sq(z-1)+sq(z-2))*.5,0.,1.);}
}
int main(){try{
 unsigned cases=0;auto source=corner();
 std::array<int,9> expected{};for(int y=0;y<3;++y)for(int x=0;x<3;++x)expected[(2-y)*3+2-x]=int(std::lround(16+224*(1-cdf(3.-x-y))));
 for(int r=0;r<4;++r){
  uint32_t a[9],b[9];check(block(source,a)==1,"general two-surface corner missing");check(block(source,b,true)==1,"hinted corner missing");check(std::equal(a,a+9,b),"clean geometric corner depends on hints");
  for(int i=0;i<9;++i)for(int shift:{0,8,16})check(std::abs(int((a[i]>>shift)&255)-expected[i])<=2,"corner differs from independent area oracle");
  source=rotate(source);auto e=expected;for(int y=0;y<3;++y)for(int x=0;x<3;++x)e[x*3+2-y]=expected[y*3+x];expected=e;++cases;
 }
 // Every foreign RGB location in the actual support must veto this anchor.
 for(int y=C-SUPPORT_RADIUS;y<=C+SUPPORT_RADIUS;++y)for(int x=C-SUPPORT_RADIUS;x<=C+SUPPORT_RADIUS;++x){auto s=corner();put(s,x,y,{255,0,255});for(int r=0;r<4;++r){uint32_t out[9];check(block(s,out)==0,"foreign-colour corner accepted");s=rotate(s);++cases;}}
 for(int pattern=0;pattern<5;++pattern){Image s(N*N*3);for(int y=0;y<N;++y)for(int x=0;x<N;++x){int q=pattern==0?(((x+y)&1)?240:16):pattern==1?16+x*16:pattern==2?((x<=C&&y<=C)?240:16):pattern==3?80:((x<=C&&y<=C)?104:80);put(s,x,y,{q,q,q});}if(pattern==3)put(s,C,C,{128,128,128});if(pattern==4)put(s,C,C,{92,92,92});lut::Frame f;uint32_t out[9];block(s,out,false,&f);check(std::none_of(f.ufo_rules.begin(),f.ufo_rules.end(),[](uint8_t c){return c!=0;}),"checker/gradient/binary/isolated/low-contrast art reclassified as AA");++cases;}
 // A square degree-three junction has explicit two-cell arm continuation.
 Image join(N*N*3,16);for(int d=-2;d<=2;++d)put(join,C+d,C,{240,240,240});for(int d=1;d<=2;++d)put(join,C,C+d,{240,240,240});
 for(int r=0;r<4;++r){lut::Frame f;uint32_t out[9];check(block(join,out,false,&f)==2,"general square junction missing");check(f.ufo_rules[C*N+C]==5,"junction was not a source-copy recipe");check(std::all_of(out,out+9,[](uint32_t c){return c==0xf0f0f0;}),"junction changed source surface");join=rotate(join);++cases;}
 // General AA must never manufacture hinted R/notch/bowl stencil codes.
 uint32_t rng=0x53a719u;
 for(int seed=0;seed<64;++seed){Image s(N*N*3);for(auto &v:s){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;v=uint8_t(rng);}lut::Frame f;uint32_t out[9];block(s,out,false,&f);for(auto c:f.ufo_rules)check(c>=144 || (!(c&64) && (c&15)!=6 && (c&15)!=7 && (c&15)!=8 && c!=133),"glyph-only stencil escaped its hint scope");++cases;}
 std::cout<<cases<<" general-AA area, rotation, colour, texture, junction and glyph-isolation controls passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
