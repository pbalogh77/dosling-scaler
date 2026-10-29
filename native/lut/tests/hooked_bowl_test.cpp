#include "model.hpp"
#include <iostream>
#include <array>

namespace {
std::array<int,2> point(int u,int v,int k) {
  if(k&4)u=-u;
  for(int i=0;i<(k&3);++i){const int t=u;u=-v;v=t;}
  return {u,v};
}
constexpr char shape[5][6]={"####.","...##","#####","#..##","###.#"};
constexpr int targets[4][2]={{-2,-2},{-1,1},{0,1},{1,1}};
bool check(int size,int k,uint32_t fg,uint32_t bg,int edit=-1,bool hint=true,int missing=-1) {
  const int c=size/2;
  std::vector<uint8_t> src(size_t(size)*size*3);
  auto put=[&](int u,int v,uint32_t value){auto p=point(u,v,k);const int x=c+p[0],y=c+p[1];if(x<0||y<0||x>=size||y>=size)return;auto at=&src[(size_t(y)*size+x)*3];at[0]=value>>16;at[1]=value>>8;at[2]=value;};
  for(int v=-c;v<size-c;++v)for(int u=-c;u<size-c;++u)put(u,v,bg);
  for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u)put(u,v,shape[v+2][u+2]=='#'?fg:bg);
  if(edit>=0 && edit<25){const int u=edit%5-2,v=edit/5-2;put(u,v,shape[v+2][u+2]=='#'?bg:fg);}
  if(edit==25)put(-3,-2,fg); // an upper bar continuing past the alleged tip
  if(edit==26)put(-2,-3,fg); // a different surface above the tip
  if(edit==27)put(0,-1,0x0000ff); // third color in the hook
  lut::Frame f;dl_rect region{0,0,size,size};
  f.prepare(src.data(),size,size,size_t(size)*3,hint?&region:nullptr,hint?1:0,true);
  if(missing>=0) {
    const int u=missing<25?missing%5-2:(missing-25)%3-3;
    const int v=missing<25?missing/5-2:(missing-25)/3-3;
    const auto p=point(u,v,k);const int x=c+p[0],y=c+p[1];
    if(x>=0 && y>=0 && x<size && y<size)f.text[size_t(y)*size+x]=0;
  }
  lut::prepare_ufo(f);
  const bool accepted=edit<0 && missing<0 && hint && size>=7 && lut::distance(fg,bg)>=3600;
  unsigned count=0;for(auto r:f.ufo_rules)count+=(r&64)!=0;
  if(!accepted)return count==0;
  if(count!=4)return false;
  for(int role=0;role<4;++role){
    auto q=point(targets[role][0],targets[role][1],k);const int x=c+q[0],y=c+q[1];
    if(f.ufo_rules[size_t(y)*size+x]!=(64|(k<<2)|role))return false;
    uint32_t core[9],out[9];f.core(x,y,core);if(lut::ufo_block(f,x,y,core,out)!=(role==3?2:1))return false;
    for(int v=-1;v<=1;++v)for(int u=-1;u<=1;++u){
      constexpr int cap[9]={64,224,256,256,256,256,64,224,256};
      const int a=role==3?256:role?(std::abs(v)==1 && u==(role==1?-1:1)?96:0):cap[(v+1)*3+u+1];
      auto pixel=point(u,v,k);const auto value=out[(pixel[1]+1)*3+pixel[0]+1];
      uint32_t want=0;
      for(int shift:{0,8,16})want|=((((bg>>shift)&255)*(256-a)+((fg>>shift)&255)*a+128)/256)<<shift;
      if(value!=want)return false;
    }
  }
  // Input remains untouched by preparation and rendering.
  for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u){auto p=point(u,v,k);if(f.pixels[size_t(c+p[1]+2)*f.pitch+c+p[0]+2]!=(shape[v+2][u+2]=='#'?fg:bg))return false;}
  return true;
}
}
int main(){
  unsigned cases=0;
  for(int size:{5,7,9,17})for(int k=0;k<8;++k)for(auto colors:std::array<std::array<uint32_t,2>,3>{{{{0xa8a8a8,0}},{{0,0xffffff}},{{0xe09040,0x183858}}}}){
    if(!check(size,k,colors[0],colors[1]))return 1;
    if(!check(size,k,colors[0],colors[1],-1,false))return 2;
    for(int edit=0;edit<28;++edit)if(!check(size,k,colors[0],colors[1],edit))return 3;
    for(int missing=0;missing<34;++missing)if(!check(size,k,colors[0],colors[1],-1,true,missing))return 5;
    cases+=64;
  }
  for(int k=0;k<8;++k)if(!check(9,k,0x141414,0))return 4;
  std::cout<<"Hooked bowl: "<<cases+8<<" bounds, D4, palette, exact witnesses and scope cases passed\n";
}
