#include "model.hpp"
#include <iostream>

int main() {
  // Sanitizer coverage exercises the real rule preparation, its source-copy
  // block and all rotations, including the smallest actual 5x5 neighborhood.
  for (int size : {5,6,7,17}) for (int rotation=0;rotation<4;++rotation)
    for (int inverted=0;inverted<2;++inverted) {
      const uint8_t bg=inverted?224:16,fg=inverted?16:224;
      std::vector<uint8_t> src(size_t(size)*size*3,bg);
      const int cx=size/2,cy=size/2;
      auto point=[&](int u,int v) {
        for (int r=0;r<rotation;++r) {int t=u;u=-v;v=t;}
        return std::array<int,2>{cx+u,cy+v};
      };
      auto put=[&](int u,int v) {
        auto p=point(u,v);
        for (int c=0;c<3;++c) src[(size_t(p[1])*size+p[0])*3+c]=fg;
      };
      put(-1,0);put(0,0);put(1,0);put(0,1);put(1,2);
      const auto original=src;
      lut::Frame frame;const dl_rect hint{0,0,size,size};
      frame.prepare(src.data(),size,size,size_t(size)*3,&hint,1,true);
      lut::prepare_ufo(frame);
      for (int v=0;v<=1;++v) for (int u=-1;u<=1;++u) {
        auto p=point(u,v);
        if (frame.ufo_rules[size_t(p[1])*size+p[0]]!=5) return 1;
        uint32_t core[9],out[9];frame.core(p[0],p[1],core);
        if (lut::ufo_block(frame,p[0],p[1],core,out)!=2) return 2;
        for (auto value : out) if (value!=core[4]) return 3;
      }
      if (original!=src) return 4;
      frame.prepare(src.data(),size,size,size_t(size)*3,nullptr,0,true);
      lut::prepare_ufo(frame);
      for (auto rule : frame.ufo_rules) if (rule) return 5;
    }
  std::cout << "Small-letter serif: actual bounds, rotations, polarity, source-copy and no-hint scope passed\n";
}
