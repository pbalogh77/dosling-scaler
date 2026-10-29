#include "model.hpp"
#include <iostream>
namespace {
constexpr uint8_t INK=200,BG=20;
void stamp(std::vector<uint8_t> &s,int w,int x,int y,int rotation) {
  for (int v=-2;v<=2;++v) for (int u=-2;u<=1;++u) {
    const bool ink=(!v && u<0)||(u==0 && std::abs(v)==1)||
                   (u==1 && std::abs(v)==2);
    int dx=u,dy=v;
    for (int r=0;r<rotation;++r) {const int old=dx;dx=-dy;dy=old;}
    for (int c=0;c<3;++c) s[(size_t(y+dy)*w+x+dx)*3+c]=ink?INK:BG;
  }
}
void inspect(int count,bool marked=true,bool disjoint=false) {
  const int w=count*2+40,h=17;
  std::vector<uint8_t> s(size_t(w)*h*3,BG);
  for (int i=0;i<count;++i)stamp(s,w,8+i*(disjoint?12:2),8,i%2?3:1);
  lut::Frame f;dl_rect region{0,0,w,h};
  f.prepare(s.data(),w,h,size_t(w)*3,marked?&region:nullptr,marked?1:0,true);
  lut::prepare_ufo(f);
  size_t rotated=0;
  for (auto rule:f.ufo_rules) {
    if (rule&0x40)throw std::runtime_error("Temporary candidate marker leaked");
    rotated+=bool(rule&0x30);
  }
  const size_t expected=marked?(count==1||disjoint?size_t(count)*3:0):0;
  if (rotated!=expected)throw std::runtime_error("Incomplete or ambiguous notch claims");
}
void inspect_border(int w,int h,int x,int y,int separation=2) {
  std::vector<uint8_t> s(size_t(w)*h*3,BG);
  stamp(s,w,x,y,1);stamp(s,w,x+separation,y,3);
  lut::Frame f;dl_rect region{0,0,w,h};
  f.prepare(s.data(),w,h,size_t(w)*3,&region,1,true);
  lut::prepare_ufo(f);
  size_t rotated=0;
  for (auto rule:f.ufo_rules) {
    if (rule&0x40)throw std::runtime_error("Border marker cleanup failed");
    rotated+=bool(rule&0x30);
  }
  if (rotated!=(separation>2?6u:0u))
    throw std::runtime_error("Incorrect border conflict decision");
}
}
int main() {
  try {
    for (int count:{1,2,3,8,28}) {inspect(count);inspect(count,false);}
    inspect(2,true,true);
    inspect_border(7,5,2,2);
    inspect_border(11,9,6,6);
    inspect_border(11,9,2,6,6); // No early conflict: scan the last flag byte.
    std::cout<<"Notch masks/temporary-marker cleanup and isolated/ambiguous claims pass\n";
    return 0;
  } catch (const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
