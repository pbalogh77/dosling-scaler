#include "model.hpp"
#include <cmath>
#include <iostream>
using Pixels=std::vector<uint8_t>;
Pixels source() {
 Pixels s(9*9*3);
 for(int y=0;y<=4;++y)for(int x=0;x<=4;++x)
  for(int c=0;c<3;++c)s[(y*9+x)*3+c]=232;
 for(int c=0;c<3;++c)s[(4*9+4)*3+c]=116;
 s[(4*9+3)*3+1]=240;s[(3*9+4)*3+1]=224;
 return s;
}
Pixels rotate(const Pixels&s,int r) {
 Pixels d=s;
 for(int y=0;y<9;++y)for(int x=0;x<9;++x) {
  int xx=x,yy=y;for(int k=0;k<r;++k){int t=xx;xx=8-yy;yy=t;}
  for(int c=0;c<3;++c)d[(yy*9+xx)*3+c]=s[(y*9+x)*3+c];
 }
 return d;
}
int render(const Pixels&s,uint32_t*out,uint32_t*core) {
 lut::Frame f;dl_rect region{0,0,9,9};f.prepare(s.data(),9,9,27,&region,1);
 lut::prepare_ufo(f);f.core(4,4,core);return lut::ufo_block(f,4,4,core,out);
}
double cdf(double z) {
 auto sq=[](double a){return a>0?a*a:0;};
 return std::clamp((sq(z)-2*sq(z-1)+sq(z-2))*.5,0.,1.);
}
int main() {
 const int points[][2]={{-2,0},{0,-2},{-1,-1},{-2,-1},{-1,-2},{-2,-2}};
 unsigned checked=0;
 for(int r=0;r<4;++r)for(int side:{-1,1})for(int axis=0;axis<3;++axis)
  for(int first=0;first<6;++first)for(int second=first;second<6;++second) {
   auto s=source();s[((4+side*points[first][1])*9+4+side*points[first][0])*3+axis]+=16;
   if(second!=first)s[((4+side*points[second][1])*9+4+side*points[second][0])*3+axis]+=16;
   uint32_t out[9],core[9];if(render(rotate(s,r),out,core)!=1)return 1;
   for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
    int xx=2-x,yy=2-y;for(int k=0;k<r;++k){int t=xx;xx=2-yy;yy=t;}
    auto c=lut::rgb(out[yy*3+xx]);double expected=232*(1-cdf(3.-x-y));
    if(c.r!=c.g || c.r!=c.b || std::abs(c.r-expected)>2)return 2;
   }
   // A learned operator has no source continuation witnesses to justify this.
   for(int k=1;k<=4;++k)if(lut::aa_block(k,core,out,true))return 3;
   ++checked;
  }
 for(int r=0;r<4;++r)for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
  auto s=source();auto p=&s[((4+dy)*9+4+dx)*3];p[0]=255;p[1]=0;p[2]=255;
  uint32_t out[9],core[9];if(render(rotate(s,r),out,core)==1)return 4;
 }
 std::cout<<checked<<"continued-endpoint outlier/rotation/independent-area cases; foreign colors and learned-operator isolation pass\n";
}
