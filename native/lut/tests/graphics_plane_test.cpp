#include "dos_lut.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc,char **argv) {
  if (argc!=2) return 1;
  auto context=dl_create(argv[1]);
  if (!context) return 2;
  // Independently derived area-CDF weights, rounded to the existing4/64grid.
  constexpr int coverage[6][9]={
    {0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,4,20},
    {0,0,0,0,4,20,44,60,64}, {0,4,20,44,60,64,64,64,64},
    {44,60,64,64,64,64,64,64,64}, {64,64,64,64,64,64,64,64,64}};
  constexpr int colors[4][2][3]={
    {{64,64,64},{80,80,80}},{{0,0,0},{255,255,255}},
    {{224,24,8},{16,32,40}},{{0,170,170},{85,85,85}}};
  for (auto &pair : colors)
    for (int phase=-3;phase<=2;++phase) {
      constexpr int w=15,h=15,ss=w*3+5,ds=w*9+7;
      std::vector<uint8_t> source(ss*h,0xa9),out(ds*h*3,0xcd),reference;
      for (int y=0;y<h;++y)
        for (int x=0;x<w;++x)
          for (int c=0;c<3;++c)
            source[y*ss+x*3+c]=uint8_t(pair[x+3*y>=28-phase][c]);
      const auto original=source;
      for (unsigned workers : {1u,4u}) {
        if (dl_set_workers(context,workers) ||
            dl_scale3(context,source.data(),w,h,ss,out.data(),ds,nullptr,0,nullptr)) return 3;
        if (source!=original) return 4;
        for (int y=0;y<3;++y)
          for (int x=0;x<3;++x) {
            int q=coverage[phase+3][y*3+x];
            for (int c=0;c<3;++c) {
              int want=(pair[0][c]*(64-q)+pair[1][c]*q+32)/64;
              if (out[(21+y)*ds+(21+x)*3+c]!=want) return 5;
            }
          }
        for (int y=0;y<h*3;++y)
          for (int x=w*9;x<ds;++x) if (out[y*ds+x]!=0xcd) return 6;
        if (workers==1) reference=out;
        else if (reference!=out) return 7;
      }
    }
  dl_destroy(context);
  std::cout << "Source-color planes: phases, colors, workers, input/output guards passed\n";
}
