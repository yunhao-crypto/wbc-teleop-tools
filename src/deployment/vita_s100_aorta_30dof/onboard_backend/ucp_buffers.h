#pragma once
#include <hobot/dnn/hb_dnn.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
namespace vita_ucp {
inline void require(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
inline void check(int n,const char* s){require(n==0,std::string(s)+" rc="+std::to_string(n));}
#define CK(x) check((x),#x)
struct Segment{size_t logical,physical,bytes;};
struct Layout{std::vector<Segment> segments;size_t bytes=0;};
// Validate static row-major byte strides (padding allowed), then coalesce copies.
static Layout layout(const hbDNNTensorProperties& p,const std::vector<int>& shape,int type,size_t elem){
 require(p.tensorType==type && p.quantiType==NONE,"Unexpected IO dtype/quantization");
 require(p.validShape.numDimensions==int(shape.size()),"Unexpected tensor rank");
 require(p.alignedByteSize>0,"Dynamic/invalid alignedByteSize unsupported");
 size_t count=1,span=elem;
 for(int d=int(shape.size())-1;d>=0;--d){
  require(p.validShape.dimensionSize[d]==shape[d] && shape[d]>0,"Unexpected/dynamic shape");
  require(p.stride[d]>0 && size_t(p.stride[d])>=span,"Invalid/overlapping or non-row-major byte stride");
  require(size_t(shape[d]-1)<=(std::numeric_limits<size_t>::max()-span)/size_t(p.stride[d]),"Stride overflow");
  span+=size_t(shape[d]-1)*size_t(p.stride[d]);count*=size_t(shape[d]);
 }
 require(span<=size_t(p.alignedByteSize),"Tensor strides exceed allocation");
 Layout l;l.bytes=count*elem;
 for(size_t i=0;i<count;++i){size_t q=i,offset=0;for(int d=int(shape.size())-1;d>=0;--d){offset+=(q%shape[d])*size_t(p.stride[d]);q/=shape[d];}
  if(!l.segments.empty() && l.segments.back().physical+l.segments.back().bytes==offset)l.segments.back().bytes+=elem;
  else l.segments.push_back({i*elem,offset,elem});}
 return l;
}
static void pack(hbDNNTensor& t,const Layout& l,const void* src){for(auto s:l.segments)std::memcpy(static_cast<char*>(t.sysMem.virAddr)+s.physical,static_cast<const char*>(src)+s.logical,s.bytes);}
static void unpack(const hbDNNTensor& t,const Layout& l,void* dst){for(auto s:l.segments)std::memcpy(static_cast<char*>(dst)+s.logical,static_cast<const char*>(t.sysMem.virAddr)+s.physical,s.bytes);}
struct Session{
 hbDNNPackedHandle_t packed=nullptr;hbDNNHandle_t model=nullptr;std::vector<hbDNNTensor> inputs,outputs;std::vector<std::string> inames,onames;
 ~Session(){for(auto& t:inputs)if(t.sysMem.virAddr)hbUCPFree(&t.sysMem);for(auto& t:outputs)if(t.sysMem.virAddr)hbUCPFree(&t.sysMem);if(packed)hbDNNRelease(packed);}
 void load(const std::string& path){const char* p=path.c_str();CK(hbDNNInitializeFromFiles(&packed,&p,1));const char** names=nullptr;int32_t n=0;CK(hbDNNGetModelNameList(&names,&n,packed));require(n==1,"HBM must contain exactly one complete model");CK(hbDNNGetModelHandle(&model,packed,names[0]));
  int32_t cores=0;CK(hbDNNGetCompileBpuCoreNum(&cores,model));require(cores==1,"Expected single-core S100 BPU model");
  for(int output=0;output<2;++output){int32_t count=0;if(output)CK(hbDNNGetOutputCount(&count,model));else CK(hbDNNGetInputCount(&count,model));require(count>0 && count<100,"Invalid IO count");auto& ts=output?outputs:inputs;auto& ns=output?onames:inames;ts.resize(count);
   for(int i=0;i<count;++i){const char* name=nullptr;auto& t=ts[i];if(output){CK(hbDNNGetOutputName(&name,model,i));CK(hbDNNGetOutputTensorProperties(&t.properties,model,i));}else{CK(hbDNNGetInputName(&name,model,i));CK(hbDNNGetInputTensorProperties(&t.properties,model,i));}require(name,"Missing IO name");ns.emplace_back(name);
    require(t.properties.alignedByteSize>0,"Dynamic allocation unsupported");CK(hbUCPMallocCached(&t.sysMem,t.properties.alignedByteSize,0));std::memset(t.sysMem.virAddr,0,t.sysMem.memSize);CK(hbUCPMemFlush(&t.sysMem,HB_SYS_MEM_CACHE_CLEAN));
   }
  }
 }
};
static size_t index(const std::vector<std::string>& ns,const std::string& name){auto i=std::find(ns.begin(),ns.end(),name);require(i!=ns.end(),"Missing tensor: "+name);require(std::count(ns.begin(),ns.end(),name)==1,"Duplicate tensor name");return size_t(i-ns.begin());}

static void pack_selected_rows(hbDNNTensor& target,const Layout& layout,
                               const std::vector<float>& pool,size_t row_elements,
                               const int32_t ids[2]){
 require(row_elements>0 && row_elements<=std::numeric_limits<size_t>::max()/4096,"Invalid expert row size");
 const size_t row_bytes=row_elements*sizeof(float),pool_bytes=pool.size()*sizeof(float);
 require(pool.size()==1024*row_elements && layout.bytes==2*row_bytes,"Expert pool/layout size mismatch");
 require(ids[0]>=0 && ids[0]<1024 && ids[1]>=0 && ids[1]<1024,"Router expert ID out of range");
 require(target.sysMem.virAddr!=nullptr,"Missing UCP expert input memory");
 const char* source=reinterpret_cast<const char*>(pool.data());char* destination=static_cast<char*>(target.sysMem.virAddr);
 for(const auto& segment:layout.segments){
  require(segment.logical<=layout.bytes && segment.bytes<=layout.bytes-segment.logical,"Logical segment outside selected rows");
  require(segment.physical<=target.sysMem.memSize && segment.bytes<=target.sysMem.memSize-segment.physical,"Physical segment outside UCP allocation");
  size_t logical=segment.logical,physical=segment.physical,left=segment.bytes;
  while(left){
   const size_t selected=logical/row_bytes,within=logical%row_bytes;
   require(selected<2,"Invalid selected expert row");const size_t count=std::min(left,row_bytes-within);
   const size_t source_offset=size_t(ids[selected])*row_bytes+within;
   require(source_offset<=pool_bytes && count<=pool_bytes-source_offset,"Expert source read outside full pool");
   std::memcpy(destination+physical,source+source_offset,count);
   logical+=count;physical+=count;left-=count;
  }
 }
}

#undef CK
}
