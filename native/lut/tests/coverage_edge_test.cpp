// Dosling scaler: measured AA area, orientation and unrelated-material vetoes.
#include "model.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace {
constexpr int N=13,C=6;using Image=std::vector<uint8_t>;
const lut::Color bg{8,16,32},fg{200,176,224};
void check(bool b,const char*w){if(!b)throw std::runtime_error(w);}
void put(Image&s,int x,int y,lut::Color c){size_t i=(y*N+x)*3;s[i]=uint8_t(c.r);s[i+1]=uint8_t(c.g);s[i+2]=uint8_t(c.b);}
lut::Color mix(int q){return {bg.r+lut::floor_div((fg.r-bg.r)*q+128,256),bg.g+lut::floor_div((fg.g-bg.g)*q+128,256),bg.b+lut::floor_div((fg.b-bg.b)*q+128,256)};}
Image blank(){Image s(N*N*3);for(int y=0;y<N;++y)for(int x=0;x<N;++x)put(s,x,y,bg);return s;}
Image turn(const Image&s){Image a=s;for(int y=0;y<N;++y)for(int x=0;x<N;++x)for(int k=0;k<3;++k)a[(x*N+N-1-y)*3+k]=s[(y*N+x)*3+k];return a;}
std::array<uint32_t,9> turn(std::array<uint32_t,9>s){auto a=s;for(int y=0;y<3;++y)for(int x=0;x<3;++x)a[x*3+2-y]=s[y*3+x];return a;}
int block(const Image&s,std::array<uint32_t,9>&out,bool hint=false){lut::Frame f;dl_rect h{C,C,C+1,C+1};f.prepare(s.data(),N,N,N*3,hint?&h:nullptr,hint?1:0,true);lut::prepare_ufo(f);uint32_t core[9];f.core(C,C,core);lut::ufo_block(f,C,C,core,out.data());check(std::count(f.text.begin(),f.text.end(),uint8_t(1))==(hint?1:0),"coverage mutated caller hint mask");return f.ufo_rules[C*N+C];}
double cdf(double z){auto sq=[](double a){return a>0?a*a:0.;};return std::clamp((sq(z)-2*sq(z-1)+sq(z-2))*.5,0.,1.);}
std::array<uint32_t,9> oracle(int q,bool stripe){std::array<uint32_t,9>a{};double alpha=q/256.,t=alpha<=.5?std::sqrt(2*alpha):2-std::sqrt(2*(1-alpha));for(int y=0;y<3;++y)for(int x=0;x<3;++x){double v=stripe?std::clamp(3*alpha-(2-y),0.,1.):cdf(3*t-(2-x)-(2-y));lut::Color c{int(std::lround(bg.r+(fg.r-bg.r)*v)),int(std::lround(bg.g+(fg.g-bg.g)*v)),int(std::lround(bg.b+(fg.b-bg.b)*v))};a[y*3+x]=lut::packed(c);}return a;}
void compare(const std::array<uint32_t,9>&a,const std::array<uint32_t,9>&b){for(int i=0;i<9;++i)for(int k:{0,8,16})check(std::abs(int(a[i]>>k&255)-int(b[i]>>k&255))<=3,"measured AA differs from independent area oracle");}
Image fringe(int q){auto s=blank();put(s,C,C,mix(q));put(s,C+1,C,mix(q));put(s,C,C+1,mix(q));put(s,C+1,C+1,fg);put(s,C+2,C+1,fg);put(s,C+1,C+2,fg);return s;}
}
int main(){try{unsigned cases=0;
for(int family=0;family<3;++family)for(int q:{64,96,128,160,192,256})for(bool invert:{false,true}){
 if((family==2 && q>128) || (q==256 && family!=1))continue;
 auto s=blank();
 if(family==0){for(int x=-2;x<=2;++x){put(s,C+x,C,mix(q));put(s,C+x,C+1,fg);put(s,C+x,C+2,fg);}}
 if(family==1){put(s,C,C,mix(q));for(int d=1;d<=3;++d){put(s,C+d,C,fg);put(s,C,C+d,fg);}put(s,C+1,C+1,fg);put(s,C-1,C-1,mix(64));}
 if(family==2)s=fringe(q);
 auto expected=oracle(q,family==0);if(invert){for(auto&v:s)v=255-v;for(auto&v:expected)v=0xffffff-v;}
 for(int rotation=0;rotation<4;++rotation){std::array<uint32_t,9>out{};check(block(s,out)!=0,"measured AA shape not reconstructed");try{compare(out,expected);}catch(...){std::cerr<<"family "<<family<<" q "<<q<<" invert "<<invert<<" rotation "<<rotation<<"\n";for(auto v:out)std::cerr<<unsigned(v>>16)<<" ";std::cerr<<"\n";for(auto v:expected)std::cerr<<unsigned(v>>16)<<" ";std::cerr<<"\n";throw;}
  // Mean source area is conserved, within integer RGB/output rounding.
  for(int k:{0,8,16}){int sum=0;for(auto v:out)sum+=v>>k&255;const int source=s[(C*N+C)*3+2-k/8];check(std::abs(sum-9*source)<=18,"AA block changed mean coverage");}
  s=turn(s);expected=turn(expected);++cases;}
}
// Fifth RGB colour, bad ratio, low contrast, absent diagonal support, and
// ambiguous multi-arm geometry must not enter the new prepared block route.
for(int mode=0;mode<5;++mode)for(int rotation=0;rotation<4;++rotation){auto s=fringe(64);
 if(mode==0){put(s,C-2,C-2,{255,0,255});put(s,C+2,C+2,mix(160));}
 if(mode==1){auto c=mix(64);c.g+=20;put(s,C,C,c);}
 if(mode==2)for(auto&v:s)v=uint8_t(80+v/16);
 if(mode==3){put(s,C+2,C+1,bg);put(s,C+1,C+2,bg);}
 if(mode==4){put(s,C-1,C,fg);put(s,C-2,C,fg);}
 for(int i=0;i<rotation;++i)s=turn(s);std::array<uint32_t,9>out{};check(block(s,out)!=169,"unsupported fringe accepted");++cases;}
// This real dialogue palette describes brown panel surfaces plus ink, not a
// coherent RGB mixing segment. It must not grow dark triangles at the panel.
{auto s=blank();const lut::Color colors[]={{0,0,0},{80,68,48},{93,80,60},{170,157,125}};
 for(int y=0;y<N;++y)for(int x=0;x<N;++x)put(s,x,y,colors[0]);
 for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x)put(s,C+x,C+y,colors[1]);
 put(s,C,C,colors[2]);for(int d=1;d<=2;++d){put(s,C+d,C,colors[3]);put(s,C,C+d,colors[3]);}
 std::array<uint32_t,9>out{};check(block(s,out)!=169,"unrelated brown panel shades treated as AA");++cases;}
{std::array<uint32_t,9>out{};check(block(fringe(64),out,true)!=169,"coverage crossed caller hint boundary");++cases;}
std::cout<<cases<<" stripe, filled-corner, weak-fringe, area, D4/polarity and material-veto cases passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
