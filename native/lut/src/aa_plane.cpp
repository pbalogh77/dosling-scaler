// Source-supported straight two-color AA; integer preparation and rendering.
#include "model.hpp"
#include "aa_math.hpp"
namespace lut {
namespace {
using namespace aa;
uint32_t at(const Frame &f,int x,int y){return f.pixels[size_t(y+2)*f.pitch+x+2];}
Color med3(Color a,Color b,Color c){auto m=[](int x,int y,int z){return x+y+z-std::min({x,y,z})-std::max({x,y,z});};return {m(a.r,b.r,c.r),m(a.g,b.g,c.g),m(a.b,b.b,c.b)};}
bool fit(const Frame &f,int x,int y,int axis,AaPlane &out){
 auto sample=[&](int u,int v){return at(f,x+(axis?v:u),y+(axis?u:v));};
 const auto fg=packed(med3(rgb(sample(-1,-2)),rgb(sample(0,-2)),rgb(sample(1,-2))));
 const auto bg=packed(med3(rgb(sample(-1,2)),rgb(sample(0,2)),rgb(sample(1,2))));
 const auto fc=rgb(fg),bc=rgb(bg);const int dr=fc.r-bc.r,dg=fc.g-bc.g,db=fc.b-bc.b,n=dr*dr+dg*dg+db*db;
 if(n<6912 || sample(0,0)==fg || sample(0,0)==bg)return false;
 const Projection project(n);
 const int residual=144;int ratios[5][5],column[5]={},fa=0,ba=0,mid=0;
 for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u){
  const auto value=sample(u,v);int ratio;
  if(value==fg)ratio=P;
  else if(value==bg)ratio=0;
  else{
   const auto c=rgb(value);ratio=project.nearest(int64_t((c.r-bc.r)*dr+(c.g-bc.g)*dg+(c.b-bc.b)*db)*P);
   if(ratio< -P/16 || ratio>P*17/16)return false;
   const Color predicted{bc.r+int(floor64(int64_t(dr)*ratio+P/2,P)),bc.g+int(floor64(int64_t(dg)*ratio+P/2,P)),bc.b+int(floor64(int64_t(db)*ratio+P/2,P))};
   const int er=c.r-predicted.r,eg=c.g-predicted.g,eb=c.b-predicted.b;
   if(er*er+eg*eg+eb*eb>residual)return false;
  }
  ratios[v+2][u+2]=ratio;fa+=ratio>=P*15/16;ba+=ratio<=P/16;mid+=ratio>0 && ratio<P;
  column[u+2]+=clip(ratio,0,P);
 }
 if(mid<3 || fa<5 || ba<5 || ratios[2][2]<=0 || ratios[2][2]>=P)return false;
 int slope=int(nearest_even64(column[1]-column[3],2));if(std::abs(slope)>P)return false;
 int h2=2*column[2]-4*P+slope;int maximum=0,total=0;
 auto validate=[&](int ss,int hh,int &maximum,int &total){
  maximum=0;total=0;const Geometry geometry(ss,hh,3);
  for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u){
   const int error=std::abs(geometry.q8(u*3,v*3)*(P/256)-ratios[v+2][u+2]);
   if(error>4*(P/256))return false;maximum=std::max(maximum,error);total+=error;
  }
  return true;
 };
 auto complete_band=[&](int ss,int hh){const int a=hh+4*ss,b=hh-6*ss;return std::min(a,b)>=-4*P && std::max(a,b)<=6*P;};
 auto five_columns=[&](const int *c){
  const int ss=int(nearest_even64(2*c[0]+c[1]-c[3]-2*c[4],10));
  const int mean=int(floor64(c[0]+c[1]+c[2]+c[3]+c[4]+2,5));
  return std::array<int,2>{ss,2*mean-4*P+ss};
 };
 // Integrating a column measures the line only when its complete vertical
 // span is visible. Then all five columns provide a lower-noise slope and
 // phase estimate; clipped outer columns retain the central-three fit.
 bool fitted=false;
 if(complete_band(slope,h2)){
  const auto q=five_columns(column);int mm,tt;
  if(std::abs(q[0])<=P && complete_band(q[0],q[1]) && validate(q[0],q[1],mm,tt)){
   slope=q[0];h2=q[1];maximum=mm;total=tt;fitted=true;
  }
 }
 if(!fitted && !validate(slope,h2,maximum,total))return false;
 // Do not integrate endpoint-field noise as extra geometric coverage. Two
 // varying deep-field witnesses justify re-estimating from the partial band.
 // One isolated quantized edge tip is insufficient evidence of field noise.
 const Geometry observed_geometry(slope,h2,3);
 int varying=0;
 for(int v:{-2,2})for(int u=-1;u<=1;++u){
  const int q=observed_geometry.q8(u*3,v*3);
  if((q==0 && sample(u,v)!=bg) || (q==256 && sample(u,v)!=fg))++varying;
 }
 if(varying>=2){
  int observed[5]={};
  for(int v=-2;v<=2;++v)for(int u=-2;u<=2;++u){
   const int q=observed_geometry.q8(u*3,v*3);
   observed[u+2]+=q==0?0:q==256?P:clip(ratios[v+2][u+2],0,P);
  }
  const auto q=complete_band(slope,h2)?five_columns(observed):std::array<int,2>{int(nearest_even64(observed[1]-observed[3],2)),0};
  const int ss=q[0],hh=complete_band(slope,h2)?q[1]:2*observed[2]-4*P+ss;
  int mm,tt;
  if(std::abs(ss)<=P && validate(ss,hh,mm,tt)){slope=ss;h2=hh;maximum=mm;total=tt;}
 }
 out.index=size_t(y)*f.w+x;out.fg=fg;out.bg=bg;out.s=int16_t(slope);out.h2=int16_t(h2);out.error=uint16_t(maximum);out.total=uint16_t(total);out.axis=uint8_t(axis);out.eligible=true;out.reciprocal=Geometry::inverse(slope,1);return true;
}
void source_block(const AaPlane &p,int dx,int dy,uint32_t own,uint32_t *out){
 const int u0=(p.axis?dy:dx)*3,v0=(p.axis?dx:dy)*3;
 const int t=3*p.h2-2*p.s*(u0+(p.s<0?3:0))-2*P*v0;
 if(t<=0 || t>=3*(std::abs(2*p.s)+2*P)){std::fill_n(out,9,own);return;}
 const Geometry geometry(p.s,p.h2,1,p.reciprocal);
 for(int y=0;y<3;++y)for(int x=0;x<3;++x){
  const int u=p.axis?u0+y:u0+x,v=p.axis?v0+x:v0+y;
  out[y*3+x]=aa::mix(p.bg,p.fg,unsigned(geometry.q8(u,v)));
 }
 if(std::all_of(out,out+9,[&](uint32_t c){return c==p.fg;}) ||
    std::all_of(out,out+9,[&](uint32_t c){return c==p.bg;}))std::fill_n(out,9,own);
}
auto rank(const AaPlane &p){
 const int d=p.h2-P-p.s;
 return std::make_tuple(p.error,p.total,std::abs(int(p.s)),std::min(p.fg,p.bg),
                       std::max(p.fg,p.bg),std::abs(d),d*(p.fg>p.bg?1:-1));
}
struct Write {size_t index,unit;int8_t dx,dy;};
uint32_t source_at(const Frame &f,size_t index){
 return f.pixels[(index/f.w+2)*f.pitch+index%f.w+2];
}
} // namespace
void collect_aa_plane(Frame &f,int x,int y,bool hinted_only){
 AaPlane best;bool found=false;
 for(int axis=0;axis<2;++axis){
  if(distance(at(f,x-(axis?1:0),y-(axis?0:1)),at(f,x+(axis?1:0),y+(axis?0:1)))<1024)continue;
  AaPlane p;
  if(fit(f,x,y,axis,p) && (!found || rank(p)<rank(best))){best=p;found=true;}
 }
 if(!found)return;
 for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx)
  if(bool(f.text[size_t(y+dy)*f.w+x+dx])!=hinted_only)return;
 f.aa_planes.push_back(best);
}
void finish_aa_planes(Frame &f,const std::vector<size_t> &square_junctions){
 auto &planes=f.aa_planes;if(planes.empty())return;
 planes.erase(std::remove_if(planes.begin(),planes.end(),[&](const AaPlane &p){
  const int x=int(p.index%f.w),y=int(p.index/f.w);
  // An affine AA unit must not steal any block of a proven square join.
  // Sorted sparse indices avoid another persistent full-frame byte map.
  for(int dy=-2;dy<=2 && !square_junctions.empty();++dy) {
   const size_t first=size_t(y+dy)*f.w+x-2;
   const auto it=std::lower_bound(square_junctions.begin(),square_junctions.end(),first);
   if(it!=square_junctions.end() && *it<=first+4)return true;
  }
  for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx){
   const unsigned r=f.ufo_rules[size_t(y+dy)*f.w+x+dx],k=r&15;
   if(r==133 || (r&64) || (k>=6 && k<=8))return true;
  }
  return false;
 }),planes.end());
 if(planes.empty())return;
 std::vector<Write> writes;writes.reserve(planes.size()*25);
 for(size_t i=0;i<planes.size();++i)for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx)
  writes.push_back({size_t(int64_t(planes[i].index)+int64_t(dy)*f.w+dx),i,int8_t(dx),int8_t(dy)});
 std::sort(writes.begin(),writes.end(),[](const Write&a,const Write&b){return a.index<b.index;});
 // Evaluate every proposal before vetoing any, so conflicts do not propagate
 // through a greedy raster-order chain. Opaque fields preserve source RGB.
 for(size_t begin=0;begin<writes.size();){
  size_t end=begin+1;while(end<writes.size() && writes[end].index==writes[begin].index)++end;
  int lo[27],hi[27];std::fill_n(lo,27,255);std::fill_n(hi,27,0);
  const auto own=source_at(f,writes[begin].index);
  for(size_t i=begin;i<end;++i){
   const auto &v=writes[i];uint32_t pixels[9];source_block(planes[v.unit],v.dx,v.dy,own,pixels);
   for(int j=0;j<9;++j)for(int c=0;c<3;++c){const int value=(pixels[j]>>(c*8))&255;lo[j*3+c]=std::min(lo[j*3+c],value);hi[j*3+c]=std::max(hi[j*3+c],value);}
  }
  bool conflict=false;for(int j=0;j<27;++j)conflict|=hi[j]-lo[j]>8;
  if(conflict)for(size_t i=begin;i<end;++i)planes[writes[i].unit].eligible=false;
  begin=end;
 }
 for(size_t begin=0;begin<writes.size();){
  size_t end=begin+1;while(end<writes.size() && writes[end].index==writes[begin].index)++end;
  const size_t index=writes[begin].index;const auto own=source_at(f,index);
  int sums[27]={},chosen=0;size_t first=begin;
  std::tuple<int,int,int> score{999,999999,999999};uint32_t first_pixels[9]={};
  for(size_t i=begin;i<end;++i){
   const auto &v=writes[i];const auto &p=planes[v.unit];if(!p.eligible)continue;
   const auto next=std::make_tuple(int(v.dx)*v.dx+int(v.dy)*v.dy,int(p.error),int(p.total));
   if(next>score)continue;
   if(next<score){std::fill_n(sums,27,0);chosen=0;score=next;first=i;}
   uint32_t pixels[9];source_block(p,v.dx,v.dy,own,pixels);
   if(!chosen)std::copy_n(pixels,9,first_pixels);
   for(int j=0;j<9;++j)for(int c=0;c<3;++c)sums[j*3+c]+=(pixels[j]>>(8*c))&255;
   ++chosen;
  }
  if(chosen){
   std::array<uint32_t,9> pixels{};
   for(int j=0;j<9;++j)for(int c=0;c<3;++c)pixels[j]|=uint32_t((sums[j*3+c]+chosen/2)/chosen)<<(8*c);
   if(std::all_of(pixels.begin(),pixels.end(),[&](uint32_t v){return v==own;}))f.ufo_rules[index]=5;
   else if(std::equal(pixels.begin(),pixels.end(),first_pixels)){
    const auto &v=writes[first];f.ufo_rules[index]=uint8_t(144+(v.dy+2)*5+v.dx+2);
   }else{f.aa_ties.push_back({index,pixels});f.ufo_rules[index]=169;}
  }
  begin=end;
 }
 // Coordinates, not vector positions, are encoded in the byte map. Removing
 // rejected parameters keeps render lookups/cache small and remains ordered.
 planes.erase(std::remove_if(planes.begin(),planes.end(),[](const AaPlane&p){return !p.eligible;}),planes.end());
}
int aa_plane_block(const Frame &f,int x,int y,uint8_t code,uint32_t own,uint32_t *out){
 const size_t index=size_t(y)*f.w+x;
 if(code==169){
  const auto p=std::lower_bound(f.aa_ties.begin(),f.aa_ties.end(),index,[](const AaTie&a,size_t i){return a.index<i;});
  if(p==f.aa_ties.end() || p->index!=index)return 0;
  std::copy(p->pixels.begin(),p->pixels.end(),out);return 1;
 }
 if(code>168)return 0;
 const int offset=code-144,dx=offset%5-2,dy=offset/5-2;
 const size_t anchor=size_t(y-dy)*f.w+x-dx;
 const auto p=std::lower_bound(f.aa_planes.begin(),f.aa_planes.end(),anchor,[](const AaPlane&a,size_t i){return a.index<i;});
 if(p==f.aa_planes.end() || p->index!=anchor)return 0;
 source_block(*p,dx,dy,own,out);return 1;
}
} // namespace lut
