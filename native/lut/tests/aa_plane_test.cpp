#include "model.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace {
using Image=std::vector<uint8_t>;
void check(bool v,const char *s){if(!v)throw std::runtime_error(s);}
// Independent continuous cell-area oracle, used only to construct test inputs.
long double area(int p,int q,long double t,int step){
 if(!p)return std::max(0.L,std::min(1.L,t/(q*step)));
 t=std::max(0.L,std::min(t,static_cast<long double>((p+q)*step)));
 auto square=[](long double a){return a>0?a*a:0.L;};
 return (square(t)-square(t-p*step)-square(t-q*step)+square(t-(p+q)*step))/(2*p*q*step*step);
}
Image source(int w,int h,int p,int q,int phase){
 Image s(size_t(w)*h*3);
 const long double c=(w/2)*768*(p+q)+phase*3*(p+q);
 for(int y=0;y<h;++y)for(int x=0;x<w;++x){
  const auto a=area(p,q,c-p*x*768-q*y*768,768);
  for(int k=0;k<3;++k)s[(size_t(y)*w+x)*3+k]=uint8_t(std::floor(16+224*a+0.5L));
 }
 return s;
}
Image turn(const Image&s,int n,bool mirror,int r){
 Image a=s;
 if(mirror)for(int y=0;y<n;++y)for(int x=0;x<n;++x)for(int c=0;c<3;++c)a[(size_t(y)*n+x)*3+c]=s[(size_t(y)*n+n-1-x)*3+c];
 while(r--){Image b=a;for(int y=0;y<n;++y)for(int x=0;x<n;++x)for(int c=0;c<3;++c)b[(size_t(x)*n+n-1-y)*3+c]=a[(size_t(y)*n+x)*3+c];a.swap(b);}
 return a;
}
Image overlay(const Image&s,int w,int h,const std::vector<dl_rect>&rs,size_t *accepted=nullptr){
 const auto untouched=s;lut::Frame f;
 f.prepare(s.data(),w,h,w*3,rs.data(),rs.size(),true);lut::prepare_ufo(f);
 if(accepted)*accepted=f.aa_planes.size();
 Image out(size_t(w)*h*27);
 for(int y=0;y<h;++y)for(int x=0;x<w;++x){
  const unsigned code=f.ufo_rules[size_t(y)*w+x];uint32_t core[9],block[9];f.core(x,y,core);std::fill_n(block,9,core[4]);
  if(code>=144){
   check(lut::ufo_block(f,x,y,core,block)==1,"missing prepared plane/tie");
  }
  for(int v=0;v<3;++v)for(int u=0;u<3;++u){const auto c=lut::rgb(block[v*3+u]);const size_t at=(size_t(y*3+v)*w*3+x*3+u)*3;out[at]=c.r;out[at+1]=c.g;out[at+2]=c.b;}
 }
 for(const auto&p:f.aa_planes){const int x=int(p.index%w),y=int(p.index/w);check(x>=2 && y>=2 && x<w-2 && y<h-2,"anchor outside support");for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u)check(bool(f.text[size_t(y+v)*w+x+u])==bool(f.text[p.index]),"mixed hint boundary support accepted");}
 check(s==untouched,"source modified");return out;
}
}
int main(){try{
 unsigned orientations=0,bounds=0;size_t accepted=0;
 // Entire new-detector overlay must be covariant, including ownership edges.
 for(bool hinted:{false,true})for(int q:{1,2,3,4,8,16,32,64})for(int phase:{16,64,128,240}){
  const std::vector<dl_rect> hints=hinted?std::vector<dl_rect>{{0,0,13,13}}:std::vector<dl_rect>{};
  const auto s=source(13,13,1,q,phase);size_t count=0;const auto ref=overlay(s,13,13,hints,&count);accepted+=count;if(!count)std::cerr<<"no planes: hinted="<<hinted<<" slope=1/"<<q<<" phase="<<phase<<"\n";check(count>0,"clean shallow plane rejected");
  for(bool m:{false,true})for(int r=0;r<4;++r){const auto a=overlay(turn(s,13,m,r),13,13,hints);check(a==turn(ref,39,m,r),"plane overlay orientation mismatch");++orientations;}
 }
 // Small dimensions, clipped/holed hints and all four borders under ASan.
 for(int w:{1,3,4,5,6,13,17,33})for(int h:{1,4,5,13,17,33})for(int mode=0;mode<4;++mode){
  const auto s=source(w,h,1,4,128);std::vector<dl_rect>rs;
  if(mode==1)rs={{0,0,w,h}};
  if(mode==2 && w>2 && h>2)rs={{1,1,w-1,h-1}};
  if(mode==3)rs={{0,0,w,h/2},{0,h/2+1,w,h},{0,h/2,w/2,h/2+1},{w/2+1,h/2,w,h/2+1}};
  size_t count=0;overlay(s,w,h,rs,&count);if(w<5 || h<5)check(count==0,"unsupported frame admitted");++bounds;
 }
 // A foreign colour in the central support must reject that anchor, rather
 // than explain it away as a different brightness mixture.
 for(int y=4;y<=8;++y)for(int x=4;x<=8;++x){auto s=source(13,13,1,4,128);const size_t i=(y*13+x)*3;s[i]=255;s[i+1]=0;s[i+2]=255;lut::Frame f;dl_rect r{0,0,13,13};f.prepare(s.data(),13,13,39,&r,1,true);lut::prepare_ufo(f);for(const auto&p:f.aa_planes)check(p.index!=84,"foreign colour admitted at centre");}
 // Moving the same support inside a larger frame must not shift its phase.
 for(int q:{1,4,64})for(int phase:{16,128,240}){
  const auto a=overlay(source(13,13,1,q,phase),13,13,{{0,0,13,13}});
  const auto b=overlay(source(25,25,1,q,phase),25,25,{{0,0,25,25}});
  for(int y=6;y<33;++y)for(int x=6;x<33;++x)for(int c=0;c<3;++c)check(a[(y*39+x)*3+c]==b[((y+18)*75+x+18)*3+c],"translated interior changed");
 }
 // A single foreign-colour edit may veto overlapping units, but must not
 // trigger an unbounded greedy conflict chain along an otherwise clean line.
 for(int q:{1,4,64}){
  auto s=source(65,65,1,q,128);const auto a=overlay(s,65,65,{{0,0,65,65}});
  s[(32*65+32)*3]=255;s[(32*65+32)*3+1]=0;s[(32*65+32)*3+2]=255;
  const auto b=overlay(s,65,65,{{0,0,65,65}});
  for(int y=0;y<195;++y)for(int x=0;x<195;++x)if(std::abs(y/3-32)>12 || std::abs(x/3-32)>12)for(int c=0;c<3;++c)check(a[(y*195+x)*3+c]==b[(y*195+x)*3+c],"unbounded conflict propagation");
 }
 std::cout<<orientations<<" clean D4 overlays, "<<bounds<<" bounded/hint cases, 25 foreign-colour anchors, 9 translations, 3 local edits; "<<accepted<<" clean units passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
