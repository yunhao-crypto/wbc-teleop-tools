// Bounds-checked selected reader for the pinned HumanoidLowState schema.
// No SDK, transport, FlatBuffers runtime, quaternion arithmetic or control API.
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <limits>

namespace {
constexpr const char* schema_hash="de514c40e02f4cd730a3154970e57f48f8f6ecbeaf0af9ed3f19b2df79bf3765";
struct Reader {
 const uint8_t* b; size_t n;
 void check(size_t p,size_t len) const {if(p>n || len>n-p) throw std::runtime_error("FlatBuffer bounds violation");}
 template<class T>T get(size_t p)const {check(p,sizeof(T));T value;std::memcpy(&value,b+p,sizeof(T));return value;}
 size_t indirect(size_t p)const {auto d=get<uint32_t>(p);if(!d)throw std::runtime_error("Null FlatBuffer offset");check(p,d);size_t target=p+d;check(target,4);return target;}
 struct Table {size_t pos,vt;uint16_t vs,os;};
 Table table(size_t p)const {
  auto delta=get<int32_t>(p);int64_t v=int64_t(p)-int64_t(delta);
  if(v<0)throw std::runtime_error("Invalid vtable offset");
  auto vt=size_t(v);auto vs=get<uint16_t>(vt);auto os=get<uint16_t>(vt+2);
  if(vs<4 || (vs&1) || os<4)throw std::runtime_error("Invalid vtable sizes");
  check(vt,vs);check(p,os);return {p,vt,vs,os};
 }
 size_t field(const Table&t,unsigned entry,size_t width)const {
  if(entry>=t.vs)return 0;
  auto off=get<uint16_t>(t.vt+entry);if(!off)return 0;
  if(off<4 || off>t.os || width>size_t(t.os-off))throw std::runtime_error("Invalid table field offset");
  check(t.pos+off,width);return t.pos+off;
 }
 template<class T>T scalar(const Table&t,unsigned entry,T fallback=T{})const {auto p=field(t,entry,sizeof(T));return p?get<T>(p):fallback;}
 size_t vector(const Table&t,unsigned entry,uint32_t&count)const {
  auto p=field(t,entry,4);if(!p)throw std::runtime_error("Missing required vector");
  auto v=indirect(p);count=get<uint32_t>(v);check(v+4,size_t(count)*4);return v+4;
 }
};
}
extern "C" const char* vita_state_schema_hash(){return schema_hash;}
// Output f32: q[30],dq[30],quat[4],gyro[3]; u64: state_id,ts_state_real.
static int decode(const uint8_t*buffer,size_t length,
 const int32_t*slots,const int32_t*roles,float*out,uint64_t*meta,char*error,size_t error_size,
 float* all_q=nullptr,uint64_t* published=nullptr){
 try {
  const uint16_t endian=1;if(*reinterpret_cast<const uint8_t*>(&endian)!=1)throw std::runtime_error("Requires little endian");
  if(!buffer||!slots||!roles||!out||!meta)throw std::runtime_error("Null decoder argument");
  Reader r{buffer,length};auto root=r.table(r.indirect(0));
  if(!r.scalar<uint8_t>(root,24))throw std::runtime_error("LowState attitude invalid");
  int map[44];for(auto &v:map)v=-1;
  for(int j=0;j<30;++j){int s=slots[j];if(s<0||s>=44||map[s]!=-1)throw std::runtime_error("Invalid policy slot map");map[s]=j;}
  uint32_t count=0;auto motors=r.vector(root,20,count);
  if(count!=44)throw std::runtime_error("Expected 44 motor slots");
  uint64_t seen=0;
  for(unsigned i=0;i<count;++i){
   auto m=r.table(r.indirect(motors+4*i));
   const auto slot=r.scalar<uint8_t>(m,30);
   if(slot>=44 || (seen&(uint64_t(1)<<slot)))throw std::runtime_error("Invalid or duplicate motor slot");
   seen|=uint64_t(1)<<slot;
   if(r.scalar<uint8_t>(m,32)!=roles[slot])throw std::runtime_error("Unexpected motor slot role");
   float q=r.scalar<float>(m,12);
   if(all_q){if(!std::isfinite(q))throw std::runtime_error("Nonfinite measured slot position");all_q[slot]=q;}
   int j=map[slot];if(j>=0){float dq=r.scalar<float>(m,14);
    if(!std::isfinite(q)||!std::isfinite(dq))throw std::runtime_error("Nonfinite policy Q/Dq");
    out[j]=q;out[30+j]=dq;}
  }
  auto imus=r.vector(root,6,count);if(count<1||count>2)throw std::runtime_error("Expected one or two IMUs");
  unsigned locations=0;bool hip=false;
  for(unsigned i=0;i<count;++i){auto sample=r.table(r.indirect(imus+4*i));auto loc=r.scalar<uint8_t>(sample,4);
   if(loc>1 || (locations&(1u<<loc)))throw std::runtime_error("Unknown or duplicate IMU");
   locations|=1u<<loc;
   auto state=r.field(sample,6,52);if(!state)throw std::runtime_error("Missing IMU state");
   if(loc==0){if(r.scalar<uint8_t>(sample,10)!=1)throw std::runtime_error("HIP IMU status not OK");hip=true;
    for(int j=0;j<7;++j){float v=r.get<float>(state+4*j);if(!std::isfinite(v))throw std::runtime_error("Nonfinite HIP quaternion/gyro");out[60+j]=v;}
   }
  }
  if(!hip)throw std::runtime_error("Missing HIP IMU");
  meta[0]=r.scalar<uint16_t>(root,10);meta[1]=r.scalar<uint64_t>(root,12);
  if(published)*published=r.scalar<uint64_t>(root,14);
  return 0;
 }catch(const std::exception&e){if(error&&error_size)std::snprintf(error,error_size,"%s",e.what());return -1;}
 catch(...){if(error&&error_size)std::snprintf(error,error_size,"Unknown decoder failure");return -1;}
}

// Preserve the policy-only ABI for recorded-state tools.
extern "C" int vita_state_decode(const uint8_t*buffer,size_t length,
 const int32_t*slots,const int32_t*roles,float*out,uint64_t*meta,char*error,size_t error_size){
 return decode(buffer,length,slots,roles,out,meta,error,error_size);
}
// Live ABI: the same pass also returns all44 Q and the source publish timestamp.
extern "C" int vita_state_decode_live(const uint8_t*buffer,size_t length,
 const int32_t*slots,const int32_t*roles,float*out,uint64_t*meta,float*all_q,
 char*error,size_t error_size){
 if(!all_q || !meta){if(error&&error_size)std::snprintf(error,error_size,"Null live output");return -1;}
 return decode(buffer,length,slots,roles,out,meta,error,error_size,all_q,meta+2);
}
