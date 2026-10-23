#include <stdexcept>
#include "ucp_buffers.h"
#include "backend.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace vita_ucp;
static hbDNNTensorProperties properties(bool padded){
 hbDNNTensorProperties p{};p.tensorType=HB_DNN_TENSOR_TYPE_F32;p.quantiType=NONE;
 p.validShape.numDimensions=3;p.validShape.dimensionSize[0]=2;p.validShape.dimensionSize[1]=2;p.validShape.dimensionSize[2]=2;
 p.stride[2]=4;p.stride[1]=8;p.stride[0]=padded?32:16;p.alignedByteSize=padded?64:32;return p;
}
int main(){
 size_t cases=0;
 std::vector<float> pool(1024*4);for(size_t i=0;i<pool.size();++i)pool[i]=float(i)+.25f;
 const int32_t selections[][2]={{0,1023},{1023,0},{17,980},{4,4}};
 for(bool padding:{false,true}){
  auto p=properties(padding);auto l=layout(p,{2,2,2},HB_DNN_TENSOR_TYPE_F32,4);
  if(!padding)assert(l.segments.size()==1&&l.segments[0].bytes==32); // crosses expert row
  for(const auto& ids:selections){
   std::array<unsigned char,64> bytes;bytes.fill(0xA5);hbDNNTensor t{};t.properties=p;t.sysMem.virAddr=bytes.data();t.sysMem.memSize=64;
   pack_selected_rows(t,l,pool,4,ids);float actual[8];unpack(t,l,actual);
   for(size_t selected=0;selected<2;++selected)for(size_t col=0;col<4;++col)assert(actual[selected*4+col]==pool[size_t(ids[selected])*4+col]);
   if(padding)for(size_t i=16;i<32;++i)assert(bytes[i]==0xA5);
   float test[8]={-0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};pack(t,l,test);unpack(t,l,actual);assert(std::memcmp(test,actual,sizeof(test))==0);++cases;
  }
 }
 bool rejected=false;auto overlap=properties(false);overlap.stride[0]=8;try{layout(overlap,{2,2,2},HB_DNN_TENSOR_TYPE_F32,4);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 rejected=false;auto p=properties(false);auto l=layout(p,{2,2,2},HB_DNN_TENSOR_TYPE_F32,4);hbDNNTensor t{};std::array<char,32> bytes{};t.sysMem.virAddr=bytes.data();t.sysMem.memSize=32;int32_t invalid[2]={1,1024};try{pack_selected_rows(t,l,pool,4,invalid);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 assert(index(std::vector<std::string>{"a","b"},"b")==1);
 std::cout<<"{\"layout_cases\":"<<cases<<",\"invalid_layout_rejected\":true,\"invalid_expert_rejected\":true,\"sizeof_config\":"<<sizeof(VitaConfig)<<",\"sizeof_state\":"<<sizeof(VitaState)<<",\"sizeof_timing\":"<<sizeof(VitaTiming)<<",\"config_timeout_offset\":"<<offsetof(VitaConfig,timeout_ms)<<",\"timing_total_offset\":"<<offsetof(VitaTiming,total_ms)<<"}\n";
}
