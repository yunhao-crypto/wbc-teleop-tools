#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace vita_router {
inline void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
template<class T>std::vector<T> read(std::ifstream& f,size_t n){std::vector<T> v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));require(bool(f),"Short router parameter file");return v;}
inline float dot(const float* x,const float* w,size_t n){
#if defined(__aarch64__)
 float32x4_t a=vdupq_n_f32(0),b=a,c=a,d=a;
 for(size_t k=0;k<n;k+=16){a=vfmaq_f32(a,vld1q_f32(x+k),vld1q_f32(w+k));b=vfmaq_f32(b,vld1q_f32(x+k+4),vld1q_f32(w+k+4));c=vfmaq_f32(c,vld1q_f32(x+k+8),vld1q_f32(w+k+8));d=vfmaq_f32(d,vld1q_f32(x+k+12),vld1q_f32(w+k+12));}
 return vaddvq_f32(vaddq_f32(vaddq_f32(a,b),vaddq_f32(c,d)));
#else
 // Host build/syntax support only: target numerical acceptance uses actual ARM.
 float lanes[16]={};for(size_t k=0;k<n;k+=16)for(size_t j=0;j<16;++j)lanes[j]=std::fma(x[k+j],w[k+j],lanes[j]);
 float q[4];for(size_t j=0;j<4;++j)q[j]=(lanes[j]+lanes[j+4])+(lanes[j+8]+lanes[j+12]);return (q[0]+q[1])+(q[2]+q[3]);
#endif
}
struct Router {
 static constexpr size_t obs_dim=3126, route_dim=3028, padded=3040;
 std::vector<int32_t> index;
 std::vector<float> mean,divisor,w0,b0,w2,b2,w3;
 std::array<float,obs_dim> normalized{};
 std::array<float,padded> x{};
 std::array<float,2048> h{};
 std::array<float,512> z{};
 std::array<float,1024> logits{},prob{};
 std::array<float,2> gates{};
 int32_t ids[2]={};
 // A persistent second worker computes independent output rows. Each dot
 // retains its original accumulation order; no floating-point reduction is split.
 std::mutex row_mutex;
 std::condition_variable row_ready,row_done;
 bool row_pending=false,row_complete=true,row_stop=false;
 std::thread row_thread;
 void hidden_rows(size_t first,size_t last){
  for(size_t j=first;j<last;++j){float v=dot(x.data(),w0.data()+j*padded,padded)+b0[j];h[j]=v*(1.f/(1.f+std::exp(-v)));}
 }
 void row_loop(){
  for(;;){
   {std::unique_lock<std::mutex> lock(row_mutex);row_ready.wait(lock,[this]{return row_stop||row_pending;});if(row_stop)return;row_pending=false;}
   hidden_rows(1024,2048);
   {std::lock_guard<std::mutex> lock(row_mutex);row_complete=true;}row_done.notify_one();
  }
 }
 ~Router(){
  {std::lock_guard<std::mutex> lock(row_mutex);row_stop=true;}row_ready.notify_one();
  if(row_thread.joinable())row_thread.join();
 }
 Router(const Router&)=delete;Router& operator=(const Router&)=delete;
 explicit Router(const char* path){
  const uint16_t endian=1;require(*reinterpret_cast<const uint8_t*>(&endian)==1,"Router pack requires little endian");
  std::ifstream f(path,std::ios::binary);require(bool(f),"Cannot open router parameters");
  char magic[8];f.read(magic,8);require(bool(f)&&std::memcmp(magic,"VROUT001",8)==0,"Invalid router pack magic/version");
  auto dims=read<uint32_t>(f,6);require(dims==std::vector<uint32_t>({obs_dim,route_dim,padded,2048,512,1024}),"Router dimensions do not match Vita ABI");
  index=read<int32_t>(f,route_dim);mean=read<float>(f,obs_dim);divisor=read<float>(f,obs_dim);
  w0=read<float>(f,2048*padded);b0=read<float>(f,2048);w2=read<float>(f,512*2048);b2=read<float>(f,512);w3=read<float>(f,1024*512);
  require(f.peek()==std::char_traits<char>::eof(),"Trailing router bytes");
  for(int32_t i:index)require(i>=0&&size_t(i)<obs_dim,"Invalid router observation index");
  for(const auto* values:{&mean,&divisor,&w0,&b0,&w2,&b2,&w3})for(float v:*values)require(std::isfinite(v),"Nonfinite router parameter");
  for(float v:divisor)require(v>0,"Nonpositive normalizer divisor");
  for(size_t row=0;row<2048;++row)for(size_t col=route_dim;col<padded;++col)require(w0[row*padded+col]==0.f,"Router padding must be zero");
  row_thread=std::thread(&Router::row_loop,this);
 }
 void run(const float* obs){
  for(size_t i=0;i<obs_dim;++i){float v=(obs[i]-mean[i])/divisor[i];normalized[i]=std::min(10.f,std::max(-10.f,v));}
  for(size_t i=0;i<route_dim;++i)x[i]=normalized[index[i]];
  {std::lock_guard<std::mutex> lock(row_mutex);row_complete=false;row_pending=true;}row_ready.notify_one();
  hidden_rows(0,1024);
  {std::unique_lock<std::mutex> lock(row_mutex);row_done.wait(lock,[this]{return row_complete;});}
  for(size_t j=0;j<512;++j)z[j]=dot(h.data(),w2.data()+j*2048,2048)+b2[j];
  for(size_t j=0;j<1024;++j)logits[j]=dot(z.data(),w3.data()+j*512,512);
  ids[0]=0;ids[1]=1;if(logits[1]>logits[0])std::swap(ids[0],ids[1]);
  for(int j=2;j<1024;++j){if(logits[j]>logits[ids[0]]){ids[1]=ids[0];ids[0]=j;}else if(logits[j]>logits[ids[1]])ids[1]=j;}
  float sum=0;for(size_t j=0;j<1024;++j){prob[j]=std::exp(logits[j]-logits[ids[0]]);sum+=prob[j];}
  float p0=prob[ids[0]]/sum,p1=prob[ids[1]]/sum,denom=std::max(p0+p1,1e-9f);gates[0]=p0/denom;gates[1]=p1/denom;
 }
};
}
