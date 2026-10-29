#pragma once
#include <algorithm>
#include <cstdint>
namespace lut::aa {
constexpr int P=4096;
inline int64_t floor64(int64_t a,int64_t b){const auto q=a/b,r=a%b;return q-(r<0);}
inline int64_t nearest_even64(int64_t a,int64_t b){
 const int64_t biased=a+b/2,q=floor64(biased,b),r=biased-q*b;
 return q-((!(b&1) && !r && (q&1))?1:0);
}
struct Projection {
 uint64_t n,m;
 explicit Projection(int value):n(value),m(((uint64_t(1)<<32)+n-1)/n){}
 int operator()(int64_t a)const{
  const uint64_t u=a<0?uint64_t(-a):uint64_t(a);
  uint64_t q=(u*m)>>32;if(q*n>u)--q;
  const auto r=u-q*n;return a<0?-int(q)-int(r!=0):int(q);
 }
 int nearest(int64_t a)const{
  const int64_t biased=a+int64_t(n)/2;const int q=(*this)(biased);
  const auto r=biased-int64_t(q)*int64_t(n);
  return q-((!(n&1) && !r && (q&1))?1:0);
 }
};
// Coordinates are output-pixel units: source cells have step3, output step1.
// den<=9*4096*2^15, A=(num*256+den/2)<2^39. With ceil(2^40/den),
// the estimated quotient overshoots by at most one and A*m fits64bits.
// Correction plus ties-to-even is exact; no floating point or per-pixel divide.
struct Geometry {
 int s,h2,step,pstep,qstep,total;
 uint32_t den,reciprocal;
 static uint32_t inverse(int s,int step){
  if(!s)return ((uint32_t(1)<<27)+step-1)/step;
  const uint32_t c=uint32_t(std::abs(s)*step*step);
  return ((uint32_t(1)<<25)+c-1)/c;
 }
 Geometry(int slope,int phase,int side,uint32_t cached=0):s(slope),h2(phase),step(side),
  pstep(std::abs(2*s)*step),qstep(2*P*step),total(pstep+qstep),
  den(s?uint32_t(std::abs(s)*step*step)<<15:uint32_t(qstep)),
  reciprocal(cached?cached:inverse(s,step)){}
 int q8(int u,int v)const{
  const int t=3*h2-2*s*(u+(s<0?step:0))-2*P*v;
  if(t<=0)return 0;if(t>=total)return 256;
  uint64_t num;
  if(!s)num=uint32_t(t);
  else{
   auto sq=[](int x){return x>0?uint64_t(x)*uint32_t(x):uint64_t(0);};
   num=sq(t)-sq(t-pstep)-sq(t-qstep);
  }
  const uint64_t a=num*256+den/2;
  uint64_t q=(a*reciprocal)>>40;if(q*den>a)--q;
  const auto r=a-q*den;return int(q-((!r && (q&1))?1:0));
 }
};
// Q8 interpolation in independent16-bit R/B lanes, then G. The weighted
// sum plus128 is<=65408 in each lane, so there is no inter-channel carry.
inline uint32_t mix(uint32_t bg,uint32_t fg,unsigned q){
 constexpr uint32_t RB=0x00ff00ff,G=0x0000ff00;
 const uint32_t rb=((bg&RB)*(256-q)+(fg&RB)*q+0x00800080u)>>8;
 const uint32_t g=((bg&G)*(256-q)+(fg&G)*q+0x00008000u)>>8;
 return (rb&RB)|(g&G);
}
}
