#include "model.hpp"
#include <iostream>
#include <stdexcept>
namespace {
constexpr int N=17, M=8;
bool use_hints=true;
using Pixels=std::array<uint8_t,N*N*3>;
int clamp(int x) {return std::max(0,std::min(256,x));}
Pixels shape(int a,int b,bool axis=false) {
 Pixels s{};
 for(int y=0;y<N;++y)for(int x=0;x<N;++x) {
  int qy=y<M?256:y==M?a:0,qx=x<M?256:x==M?b:0;
  int q=axis?qy:256-((256-qx)*(256-qy)+128)/256;
  for(int c=0;c<3;++c)s[(y*N+x)*3+c]=uint8_t((240*q+128)/256);
 }
 return s;
}
Pixels rotate(const Pixels &s) {
 Pixels r{};
 for(int y=0;y<N;++y)for(int x=0;x<N;++x)for(int c=0;c<3;++c)
  r[(x*N+N-1-y)*3+c]=s[(y*N+x)*3+c];
 return r;
}
std::array<uint32_t,9> output(const Pixels &s,int &kind,int x=M,int y=M) {
 lut::Frame f;dl_rect region{0,0,N,N};f.prepare(s.data(),N,N,N*3,use_hints?&region:nullptr,use_hints?1:0,true);lut::prepare_ufo(f);
 kind=f.ufo_rules[y*N+x];std::array<uint32_t,9> out{};uint32_t core[9];f.core(x,y,core);
 lut::ufo_block(f,x,y,core,out.data());return out;
}
void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
}
int main() {
 try {
  for(bool scope:{false,true}) {use_hints=scope;
  int cases=0;
  // Independent subpixel strip/union areas; no renderer is used to form targets.
  for(bool axis:{false,true})for(int a:{32,64,128,192,224})for(int b:{32,64,128,192,224}) {
   auto s=shape(a,b,axis);std::array<int,9> expected{};
   for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
    const int qy=clamp(a*3-y*256),qx=clamp(b*3-x*256);
    const int q=axis?qy:256-((256-qx)*(256-qy)+128)/256;
    expected[y*3+x]=(240*q+128)/256;
   }
   for(int r=0;r<4;++r) {
    int kind;const auto out=output(s,kind);
    require(axis?((kind<144 && ((kind&15)==9 || (kind&15)==10)) || (kind>=144 && kind<=169)):(kind<144 && (kind&15)==11),"supported AA geometry not admitted");
    for(int i=0;i<9;++i)for(int shift:{0,8,16})require(std::abs(int((out[i]>>shift)&255)-expected[i])<=2,"coverage differs from independent area");
    auto next=expected;for(int y=0;y<3;++y)for(int x=0;x<3;++x)next[x*3+2-y]=expected[y*3+x];expected=next;s=rotate(s);++cases;
   }
  }
  // Source-supported faint continuation: q=16 on one arm. Treating that
  // repeated mixture as an opaque field created a competing axial stencil.
  // The geometry guard must retain the two-arm union in every rotation.
  {
   constexpr int a=16,b=127,fg=24,bg=232;Pixels s{};
   for(int y=0;y<N;++y)for(int x=0;x<N;++x) {
    const int qy=y<M?256:y==M?a:0,qx=x<M?256:x==M?b:0;
    const int q=256-((256-qx)*(256-qy)+128)/256;
    for(int c=0;c<3;++c)s[(y*N+x)*3+c]=(bg*256+(fg-bg)*q+128)/256;
   }
   std::array<int,9> expected{};
   for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
    const int qy=clamp(a*3-y*256),qx=clamp(b*3-x*256),q=256-((256-qx)*(256-qy)+128)/256;
    expected[y*3+x]=(bg*256+(fg-bg)*q+128)/256;
   }
   for(int r=0;r<4;++r) {
    int kind;auto out=output(s,kind);require(kind<144 && (kind&15)==11,"faint continuation displaced the true union");
    for(int i=0;i<9;++i)for(int shift:{0,8,16})require(std::abs(int((out[i]>>shift)&255)-expected[i])<=2,"faint union area differs");
    auto next=expected;for(int y=0;y<3;++y)for(int x=0;x<3;++x)next[x*3+2-y]=expected[y*3+x];expected=next;s=rotate(s);++cases;
   }
  }
  // At low contrast, source rounding moves nominal q=16 to observed
  // q=15. The continued-mixture guard must include that rounding interval.
  for(int fg:{83,84}) {
   constexpr int a=16,b=64;Pixels s{};std::array<int,9> expected{};
   for(int y=0;y<N;++y)for(int x=0;x<N;++x) {
    const int qy=y<M?256:y==M?a:0,qx=x<M?256:x==M?b:0;
    const int q=256-((256-qx)*(256-qy)+128)/256;s[(y*N+x)*3]=(fg*q+128)/256;
   }
   for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
    const int qy=clamp(a*3-y*256),qx=clamp(b*3-x*256),q=256-((256-qx)*(256-qy)+128)/256;expected[y*3+x]=(fg*q+128)/256;
   }
   for(int r=0;r<4;++r) {
    int kind;auto out=output(s,kind);
    if(fg==83)require(kind<144 && (kind&15)<9,"below-threshold contrast admitted");
    else {
     require(kind<144 && (kind&15)==11,"rounded faint mixture displaced a low-contrast union");
     for(int i=0;i<9;++i) {require((out[i]&65535)==0,"constant channels changed");require(std::abs(int(out[i]>>16)-expected[i])<=2,"low-contrast union area differs");}
    }
    auto next=expected;for(int y=0;y<3;++y)for(int x=0;x<3;++x)next[x*3+2-y]=expected[y*3+x];expected=next;s=rotate(s);++cases;
   }
  }
  // A foreign RGB component anywhere in the actual 5x5 support vetoes the
  // central geometry, including all four orientations. No RGB-line shortcut.
  for(int y=M-2;y<=M+2;++y)for(int x=M-2;x<=M+2;++x) {
   auto s=shape(64,192);s[(y*N+x)*3]=255;s[(y*N+x)*3+1]=0;s[(y*N+x)*3+2]=255;
   for(int r=0;r<4;++r) {int kind;output(s,kind);require(kind<144 && (kind&15)<9,"foreign colour admitted as AA");s=rotate(s);++cases;}
  }
  // Pure binary corners provide no fractional coverage evidence.
  for(int a:{0,256})for(int b:{0,256}) {int kind;output(shape(a,b),kind);require(kind<144 && (kind&15)<9,"binary art reclassified");++cases;}
  // Full 5x5 hint support is required, not just the core cross.
  auto s=shape(64,192);lut::Frame f;dl_rect region{M-1,M-1,M+2,M+2};f.prepare(s.data(),N,N,N*3,&region,1,true);lut::prepare_ufo(f);
  require(f.ufo_rules[M*N+M]<144 && (f.ufo_rules[M*N+M]&15)<9,"geometry escaped its text hint");
  std::cout<<cases<<" orthogonal AA area, rotation, foreign-colour, binary and hint cases passed; hinted="<<scope<<"\n";
  }
 }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
