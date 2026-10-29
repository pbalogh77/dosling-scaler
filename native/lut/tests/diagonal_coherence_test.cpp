#include "model.hpp"
#include <iostream>

int main() {
  constexpr int w=15,h=15;
  for (int rotation=0;rotation<4;++rotation)
    for (auto palette : std::array<std::array<uint32_t,2>,4>{{
      {{0,0xffffff}},{{0xffffff,0}},{{0x002040,0xc04020}},{{0,0xaaaaaa}}}}) {
      std::vector<uint8_t> source(w*h*3);
      auto coordinate=[&](int x,int y) {
        for(int r=0;r<rotation;++r){int t=x;x=w-1-y;y=t;}
        return std::array<int,2>{x,y};
      };
      for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        auto p=coordinate(x,y);uint32_t c=palette[x+y>=14];
        source[(p[1]*w+p[0])*3]=uint8_t(c>>16);
        source[(p[1]*w+p[0])*3+1]=uint8_t(c>>8);
        source[(p[1]*w+p[0])*3+2]=uint8_t(c);
      }
      auto select=[&](const std::vector<uint8_t> &s) {
        lut::Frame frame;frame.prepare(s.data(),w,h,w*3,nullptr,0,true);
        uint32_t core[9];frame.core(7,7,core);
        return frame.bank(7,7,core,true).index;
      };
      if(select(source)!=2)return 1;
      // Every actual5x5 source cell must be one of the two observed colors.
      for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
        auto changed=source;const size_t p=size_t((7+dy)*w+7+dx)*3;
        changed[p]=17;changed[p+1]=19;changed[p+2]=23;
        if(select(changed)!=0)return 2;
      }
      // A remote palette edit cannot disable ordinary two-color geometry.
      auto remote=source;
      for(int y=0;y<4;++y)for(int x=0;x<w;++x) {
        const size_t p=size_t(y*w+x)*3;
        remote[p]=uint8_t(13+x*7);remote[p+1]=uint8_t(17+y*13);remote[p+2]=113;
      }
      if(select(remote)!=2)return 3;
    }
  for(auto xy:std::array<std::array<int,2>,4>{{{{1,2}},{{2,1}},{{13,2}},{{2,13}}}}) {
    std::vector<uint8_t> s(w*h*3);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)
      for(int c=0;c<3;++c)s[(y*w+x)*3+c]=uint8_t(x+y>=xy[0]+xy[1]?255:0);
    lut::Frame f;f.prepare(s.data(),w,h,w*3,nullptr,0,true);uint32_t core[9];f.core(xy[0],xy[1],core);
    if(f.bank(xy[0],xy[1],core,true).index!=0)return 4;
  }
  std::cout<<"Binary diagonal coherence: rotations, polarity, every5x5sample, remote palette and actual bounds passed\n";
}
