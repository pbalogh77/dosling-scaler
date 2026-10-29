#include "model.hpp"
#include <cmath>
#include <iostream>

namespace {
using Pixels=std::vector<uint8_t>;
Pixels source(int noise=16) {
  Pixels s(9*9*3);
  for(int y=0;y<9;++y)for(int x=0;x<9;++x) {
    s[(y*9+x)*3]=uint8_t(x<=4 && y<=4?240:16);
    s[(y*9+x)*3+1]=40;s[(y*9+x)*3+2]=60;
  }
  s[(4*9+4)*3]=128;s[(4*9+4)*3+1]=uint8_t(40+noise);
  return s;
}
Pixels rotate(const Pixels&s,int r) {
  Pixels d=s;
  for(int y=0;y<9;++y)for(int x=0;x<9;++x) {
    int xx=x,yy=y;
    for(int k=0;k<r;++k){int t=xx;xx=8-yy;yy=t;}
    for(int c=0;c<3;++c)d[(yy*9+xx)*3+c]=s[(y*9+x)*3+c];
  }
  return d;
}
int render(const Pixels&s,uint32_t*out,uint32_t*core) {
  lut::Frame f;dl_rect region{0,0,9,9};
  f.prepare(s.data(),9,9,27,&region,1,true);lut::prepare_ufo(f);
  f.core(4,4,core);return lut::ufo_block(f,4,4,core,out);
}
double cdf(double z) {
  auto sq=[](double a){return a>0?a*a:0;};
  return std::clamp((sq(z)-2*sq(z-1)+sq(z-2))*.5,0.,1.);
}
}
int main() {
  for(int r=0;r<4;++r) {
    uint32_t out[9],core[9];
    if(render(rotate(source(),r),out,core)!=1)return 1;
    // Independent continuous-area reference. All RGB components use one
    // ratio; green noise does not change the red coverage or tint the edge.
    for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
      const double coverage=1-cdf(3.-x-y);
      int xx=2-x,yy=2-y;
      for(int k=0;k<r;++k){int t=xx;xx=2-yy;yy=t;}
      auto c=lut::rgb(out[yy*3+xx]);
      if(std::abs(c.r-(16+224*coverage))>2 || c.g!=40 || c.b!=60)return 2;
    }
    // The source-validated hardcoded pass opts in. Learned AA operators
    // have no such geometry witness and must retain residual144 admission.
    for(int k=1;k<=4;++k)if(lut::aa_block(k,core,out,false))return 3;
    // Every foreign-color sample in the actual5x5 neighborhood vetoes this
    // new corner; a merely similar comparison key cannot authorize it.
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
      auto s=source();const int p=((4+dy)*9+4+dx)*3;
      s[p]=240;s[p+1]=0;s[p+2]=240;
      if(render(rotate(s,r),out,core)==1)return 4;
    }
    auto hole=source();for(int c=0;c<3;++c)hole[(3*9+3)*3+c]=hole[(5*9+5)*3+c];
    if(render(rotate(hole,r),out,core)==1)return 5;
    if(render(rotate(source(32),r),out,core)==1)return 6;
  }
  std::cout<<"Bounded AA: independent area/rotations, shared RGB ratio, foreign samples, counter gaps and learned-operator isolation passed\n";
}
