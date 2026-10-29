#include "model.hpp"
#include <iostream>
#include <stdexcept>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
int main() {
 try {
  lut::Frame f; f.w=f.h=5; f.pitch=9; f.pixels.resize(81);
  f.keys.resize(25); f.text.resize(25);
  constexpr int center=40;
  const int removed[]={0,90,2,5,8,16,26,33,64,74,82,88,95,123,132,160,
    222,250,1,4,32,128,10,18,72,80,91,94,122,218};
  const int retained[]={11,22,104,208,3,6,7,9,15,20,23,40,41,43,96,105,
    144,148,150,192,212,224,232,240};
  for (auto colors : {std::array<uint32_t,2>{0,0x555555},
                      std::array<uint32_t,2>{0x285064,0x286450}}) {
   auto setup=[&](int prefix) {
    std::fill(f.pixels.begin(),f.pixels.end(),colors[0]);
    int bit=0;
    for(int i=0;i<9;++i) f.pixels[center+lut::cy[i]*f.pitch+lut::cx[i]]=
      i==4?colors[0]:colors[(prefix>>bit++)&1];
    f.keys[12]=uint8_t(prefix);
   };
   uint32_t core[9];
   for(int prefix:removed) {
    setup(prefix); f.core(2,2,core);
    CHECK(f.bank(2,2,core,true).index==0);
    CHECK(f.bank(2,2,core,false).index==0);
   }
   for(int prefix:retained) {
    setup(prefix); f.core(2,2,core);
    auto selected=f.bank(2,2,core,true);
    CHECK(selected.index==2 && selected.outer_ready && selected.outer==0);
    CHECK(f.bank(2,2,core,false).index==0);
    for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
     int pos=center+y*f.pitch+x; auto old=f.pixels[pos];
     f.pixels[pos]=0xff00ff; f.core(2,2,core);
     CHECK(f.bank(2,2,core,true).index==0);
     f.pixels[pos]=old;
    }
    f.core(2,2,core); f.text[12]=1;
    CHECK(f.bank(2,2,core,true).index==1); f.text[12]=0;
   }
  }
  std::cout<<"Checker interiors/boundaries disabled; ordinary plane, third-color and text-priority controls passed\n";
  return 0;
 } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
