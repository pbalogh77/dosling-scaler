#include "model.hpp"
#include "aa_math.hpp"
#include <iostream>
static int reference(int s,int h2,int u,int v,int step){
 const int64_t p=std::abs(2*s),q=8192,t=int64_t(h2)*768-int64_t(2*s)*(u*256+(s<0?step*256:0))-q*v*256;
 if(!p)return int(std::min<int64_t>(256,std::max<int64_t>(0,lut::aa::nearest_even64(t*256,q*step*256))));
 if(t<=0)return 0;if(t>=(p+q)*step*256)return 256;
 auto sq=[](int64_t a){return a>0?a*a:int64_t(0);};
 const int64_t a=p*step*256,b=q*step*256,den=2*p*q*step*step*256*256;
 return int(lut::aa::nearest_even64((sq(t)-sq(t-a)-sq(t-b)+sq(t-a-b))*256,den));
}
int main(){
 uint32_t state=81237;auto random=[&](){state=state*1664525u+1013904223u;return state;};
 uint64_t cases=0;
 for(int s=-4096;s<=4096;++s)for(int step:{1,3}){
  for(int j=0;j<24;++j){const int h=int(random()%57345)-24576,u=int(random()%25)-12,v=int(random()%25)-12;lut::aa::Geometry g(s,h,step);if(g.q8(u,v)!=reference(s,h,u,v,step))return 1;++cases;}
 }
 for(unsigned i=0;i<1000000;++i){const auto a=random()&0xffffff,b=random()&0xffffff;const int q=int(random()%257);auto x=lut::rgb(a),y=lut::rgb(b);const auto want=lut::packed({x.r+int(lut::aa::floor64((y.r-x.r)*q+128,256)),x.g+int(lut::aa::floor64((y.g-x.g)*q+128,256)),x.b+int(lut::aa::floor64((y.b-x.b)*q+128,256))});if(lut::aa::mix(a,b,q)!=want)return 2;}
 std::cout<<cases<<" normalized reciprocal areas and1000000packed RGB mixes exact\n";
}
