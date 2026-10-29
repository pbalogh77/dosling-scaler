#include "model.hpp"
#include <iostream>

int main() {
  for (int width : {5,6,7,13})
    for (int height : {5,6,7,13}) {
      std::vector<uint8_t> pixels(size_t(width)*height*3);
      auto put=[&](int x,int y,int r,int g,int b) {
        auto p=(size_t(y)*width+x)*3;
        pixels[p]=uint8_t(r);pixels[p+1]=uint8_t(g);pixels[p+2]=uint8_t(b);
      };
      for (int y=0;y<height;++y)
        for (int x=0;x<width;++x) put(x,y,20,40,60);
      for (int y=0;y<5;++y) put(2,y,220,220,200);
      put(3,2,220,220,200);
      for (int x=3;x<5;++x) {
        put(x,1,20,60,80);put(x,3,60,30,70);
      }
      const dl_rect hints{0,0,width,height};
      lut::Frame frame;
      frame.prepare(pixels.data(),width,height,size_t(width)*3,&hints,1,true);
      lut::prepare_ufo(frame);
      for (int y : {1,3}) {
        if (frame.ufo_rules[size_t(y)*width+3]!=5) return 1;
        uint32_t core[9],out[9];int j=0;
        for (int dy=-1;dy<=1;++dy)
          for (int dx=-1;dx<=1;++dx)
            core[j++]=frame.pixels[size_t(y+dy+2)*frame.pitch+3+dx+2];
        if (lut::ufo_block(frame,3,y,core,out)!=2) return 2;
        for (auto value : out) if (value!=core[4]) return 3;
      }
      for (auto rule : frame.ufo_rules) if (rule&0x40u) return 4;
    }
  std::cout << "Short joins: minimum bounds, source-copy blocks, marker cleanup passed\n";
}
