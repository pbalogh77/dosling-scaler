#include "model.hpp"
#include <iostream>

int main() {
  for (int w : {5,6,7,12}) for (int h : {5,6,7,11}) {
    std::vector<uint8_t> src(size_t(w)*h*3);
    auto put=[&](int x,int y) {
      for (int c=0;c<3;++c) src[(size_t(y)*w+x)*3+c]=168;
    };
    for (auto p : std::array<std::array<int,2>,10>{{
         {{1,1}},{{2,1}},{{3,1}},{{4,1}},{{1,2}},{{4,2}},
         {{1,3}},{{2,3}},{{3,3}},{{4,3}}}}) put(p[0],p[1]);
    for (int rotation=0;rotation<4;++rotation) {
      int rw=rotation%2?h:w,rh=rotation%2?w:h;
      std::vector<uint8_t> rotated(size_t(rw)*rh*3);
      auto coordinate=[&](int x,int y) {
        switch(rotation) {
        case 0:return std::array<int,2>{x,y};
        case 1:return std::array<int,2>{h-1-y,x};
        case 2:return std::array<int,2>{w-1-x,h-1-y};
        default:return std::array<int,2>{y,w-1-x};
        }
      };
      for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        auto p=coordinate(x,y);
        std::copy_n(&src[(size_t(y)*w+x)*3],3,&rotated[(size_t(p[1])*rw+p[0])*3]);
      }
      const auto original=rotated;
      const dl_rect hint{0,0,rw,rh};lut::Frame frame;
      frame.prepare(rotated.data(),rw,rh,size_t(rw)*3,&hint,1,true);
      lut::prepare_ufo(frame);
      for (int x : {2,3}) {
        auto p=coordinate(x,2);
        if (frame.ufo_rules[size_t(p[1])*rw+p[0]]!=5) return 1;
        uint32_t core[9],out[9];frame.core(p[0],p[1],core);
        if (lut::ufo_block(frame,p[0],p[1],core,out)!=2) return 2;
        for (auto value : out) if (value!=core[4]) return 3;
      }
      if (original!=rotated) return 4;
      for (auto rule : frame.ufo_rules) if (rule&0x40u) return 5;
    }
  }
  std::cout << "Two-cell counters: actual bounds, rotations, source-copy and input preservation passed\n";
}
