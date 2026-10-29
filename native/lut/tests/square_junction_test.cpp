// Dosling scaler: independent source-copy oracle for orthogonal junctions.
#include "model.hpp"
#include <iostream>
#include <stdexcept>
namespace {
constexpr int N=17,C=8;using Image=std::vector<uint8_t>;
void require(bool b,const char *why){if(!b)throw std::runtime_error(why);}
void put(Image&s,int x,int y,int c){for(int k=0;k<3;++k)s[(y*N+x)*3+k]=uint8_t(c);}
Image turn(const Image&s){Image o=s;for(int y=0;y<N;++y)for(int x=0;x<N;++x)for(int k=0;k<3;++k)o[(x*N+N-1-y)*3+k]=s[(y*N+x)*3+k];return o;}
lut::Frame frame(const Image&s,bool hints){lut::Frame f;dl_rect r{0,0,N,N};f.prepare(s.data(),N,N,N*3,hints?&r:nullptr,hints?1:0,true);lut::prepare_ufo(f);return f;}
void square(const Image&s,const lut::Frame&f,int x,int y){uint32_t core[9],out[9];f.core(x,y,core);require(lut::ufo_block(f,x,y,core,out)==2,"square join block missing");const int v=s[(y*N+x)*3];const uint32_t wanted=v*0x010101u;for(auto c:out)require(c==wanted,"junction rounded or changed its own RGB");}
}
int main(){try{
unsigned cases=0;
// All thin T/+ arm lengths, rotation/polarity and hint/no-hint treatments.
for(int arms:{3,4})for(unsigned lengths=0;lengths<16;++lengths)for(bool inverse:{false,true})for(bool hints:{false,true}) {
 Image s(N*N*3,inverse?240:16);const int fg=inverse?16:240;put(s,C,C,fg);
 const int dx[]={1,0,-1,0},dy[]={0,1,0,-1};
 for(int d=0;d<arms;++d)for(int i=1;i<=1+int((lengths>>d)&1);++i)put(s,C+i*dx[d],C+i*dy[d],fg);
 for(int rotation=0;rotation<4;++rotation){const auto f=frame(s,hints);for(int y=C-1;y<=C+1;++y)for(int x=C-1;x<=C+1;++x)square(s,f,x,y);s=turn(s);++cases;}
}
// Background row shading belongs to the original surface, never a new edge.
for(bool hints:{false,true})for(int rotation=0;rotation<4;++rotation){Image s(N*N*3,48);for(int y=0;y<N;++y)for(int x=0;x<N;++x)put(s,x,y,48+((x+y)&1)*8);for(int d=-2;d<=2;++d)put(s,C+d,C,180);put(s,C,C+1,180);for(int i=0;i<rotation;++i)s=turn(s);auto f=frame(s,hints);for(int y=C-1;y<=C+1;++y)for(int x=C-1;x<=C+1;++x)square(s,f,x,y);++cases;}
// Width-two stem crossing width-one bar: both inside quadrants stay square.
for(bool hints:{false,true})for(int rotation=0;rotation<4;++rotation){Image s(N*N*3,16);for(int i=-2;i<=2;++i){put(s,C,C+i,240);put(s,C+1,C+i,240);put(s,C+i,C,240);}put(s,C+3,C,240);for(int i=0;i<rotation;++i)s=turn(s);auto f=frame(s,hints);const int targets[8][2]={{0,0},{1,0},{-1,-1},{-1,1},{2,-1},{2,1},{-1,0},{2,0}};for(auto p:targets){int x=p[0],y=p[1];for(int i=0;i<rotation;++i){int z=x;x=-y;y=z;}square(s,f,C+x,C+y);}++cases;}
// Complete rectangular unions: one/two-pixel horizontal and vertical
// strokes, both polarities, with and without the fourth arm.
for(int sw:{1,2})for(int bw:{1,2})for(bool cross:{false,true})for(bool inverse:{false,true})for(bool hints:{false,true})for(int rotation=0;rotation<4;++rotation){
 Image s(N*N*3,inverse?240:16);const int fg=inverse?16:240;
 for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x)
   if((y>=0 && y<bw) || (x>=0 && x<sw && (cross || y<0)))put(s,C+x,C+y,fg);
 for(int i=0;i<rotation;++i)s=turn(s);auto f=frame(s,hints);
 for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x)square(s,f,C+x,C+y);++cases;
}
// A short arm turning diagonally, X, L, isolated dot and low contrast are
// not T/+ evidence. These controls test the new general admission only.
for(int mode=0;mode<6;++mode)for(int rotation=0;rotation<4;++rotation){Image s(N*N*3,16);const int fg=mode==5?40:240;put(s,C,C,fg);
 if(mode==1){for(int d:{-2,-1,1,2}){put(s,C+d,C+d,fg);put(s,C+d,C-d,fg);}}
 else if(mode!=3){for(int d=-2;d<=2;++d)put(s,C+d,C,fg);put(s,C,C+1,fg);if(mode==0)put(s,C+1,C+2,fg);else if(mode==2){put(s,C-1,C,16);put(s,C-2,C,16);}else if(mode==4){auto i=((C-1)*N+C-1)*3;s[i]=255;s[i+1]=0;s[i+2]=255;}}
 for(int i=0;i<rotation;++i)s=turn(s);const auto f=frame(s,false);require(f.ufo_rules[C*N+C]!=5,"unsupported shape claimed as square junction");++cases;
}
// The source-exact core remains square beside a distinct panel bevel.
// Neither the outside row colour nor a neighboring letter defines its core.
for(bool inverse:{false,true})for(bool hints:{false,true})for(int rotation=0;rotation<4;++rotation){
 Image s(N*N*3,inverse?240:16);const int fg=inverse?16:240;
 for(int d=-1;d<=1;++d)put(s,C+d,C,fg);for(int d=1;d<=3;++d)put(s,C,C+d,fg);
 for(int d=-2;d<=2;++d)put(s,C+d,C-2,inverse?110:146);
 for(int i=0;i<rotation;++i)s=turn(s);auto f=frame(s,hints);
 for(int y=C-1;y<=C+1;++y)for(int x=C-1;x<=C+1;++x)square(s,f,x,y);++cases;
}
// Explicit hint boundaries must never be bridged by the general matcher.
{Image s(N*N*3,16);for(int d=-2;d<=2;++d){put(s,C+d,C,240);put(s,C,C+d,240);}lut::Frame f;dl_rect r{C,C,C+1,C+1};f.prepare(s.data(),N,N,N*3,&r,1,true);lut::prepare_ufo(f);require(f.ufo_rules[C*N+C]==0,"junction crossed partial hint boundary");++cases;}
std::cout<<cases<<" square T/+ source-copy, width, shading, rotation, polarity and rejection controls passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
