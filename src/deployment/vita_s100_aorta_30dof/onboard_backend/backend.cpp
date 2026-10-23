#include "backend.h"
#include "artifact_sha256.h"
#include "ort_cuts.h"
#include "router_worker.h"
#include "ucp_buffers.h"
#include <hobot/hb_ucp_status.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
struct Fault:std::runtime_error{int code;Fault(int c,const std::string& s):std::runtime_error(s),code(c){}};
void require(bool b,int c,const std::string& s){if(!b)throw Fault(c,s);}
void check(int n,const char* s){require(n==0,VITA_RUNTIME,std::string(s)+" rc="+std::to_string(n));}
double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
void finite(const float* p,size_t n,const char* label){for(size_t i=0;i<n;++i)if(!std::isfinite(p[i]))throw Fault(VITA_NONFINITE,std::string(label)+" nonfinite at "+std::to_string(i));}
void error_text(char* p,size_t n,const char* text){if(p&&n){size_t z=std::min(n-1,std::strlen(text));std::memcpy(p,text,z);p[z]=0;}}
template<class F>int guard(char* error,size_t bytes,F fn){try{fn();error_text(error,bytes,"");return VITA_OK;}catch(const Fault& e){error_text(error,bytes,e.what());return e.code;}catch(const std::exception& e){error_text(error,bytes,e.what());return VITA_RUNTIME;}catch(...){error_text(error,bytes,"Unknown native failure");return VITA_RUNTIME;}}
void artifact_spec(const VitaArtifact& f){require(f.path&&*f.path&&f.sha256,VITA_INVALID,"Artifact path and hash required");std::string h(f.sha256);require(h.size()==64&&h.find_first_not_of("0123456789abcdef")==std::string::npos,VITA_INVALID,"Expected 64 lowercase hex SHA256");}
void verify(const VitaArtifact& f){artifact_spec(f);require(artifact_hash::file(f.path)==f.sha256,VITA_INVALID,std::string("SHA256 mismatch: ")+f.path);}
std::vector<float> pool(const VitaArtifact& f,size_t width){artifact_spec(f);std::ifstream in(f.path,std::ios::binary|std::ios::ate);require(bool(in),VITA_INVALID,"Cannot open expert pool");size_t bytes=1024*width*sizeof(float);require(in.tellg()==std::streamoff(bytes),VITA_INVALID,"Full 1024-row FP32 NK pool required");std::vector<float> result(1024*width);in.seekg(0);artifact_hash::Sha256 hash;char* out=reinterpret_cast<char*>(result.data());
 for(size_t offset=0;offset<bytes;){size_t n=std::min(size_t(8*1024*1024),bytes-offset);in.read(out+offset,n);require(bool(in),VITA_INVALID,"Short expert pool read");hash.update(out+offset,n);offset+=n;}
 require(hash.finish()==f.sha256,VITA_INVALID,"Expert pool loaded-byte SHA256 mismatch (NK layout required)");finite(result.data(),result.size(),"Expert pool");return result;
}
struct VelocityOrt {
 Ort::Session session;
 std::array<float,990> observation{};std::array<float,28> actions{};
 const char* input_name="obs";const char* output_name="actions";
 std::vector<Ort::Value> inputs,outputs;
 VelocityOrt(Ort::Env& env,const char* path):session(env,path,policy_cut::options()){
  policy_cut::validate(session,{{input_name,{1,990},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}},false);
  policy_cut::validate(session,{{output_name,{1,28},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}},true);
  inputs.emplace_back(policy_cut::tensor(observation.data(),observation.size(),{1,990}));outputs.emplace_back(policy_cut::tensor(actions.data(),actions.size(),{1,28}));
 }
 void run(){session.Run(Ort::RunOptions{nullptr},&input_name,inputs.data(),1,&output_name,outputs.data(),1);}
};
struct UcpRun {
 vita_ucp::Session session;
 hbUCPTaskHandle_t task=nullptr;
 bool submission_attempted=false;
 void execute(uint32_t timeout){
  require(task==nullptr,VITA_POISONED,"Outstanding UCP task");
  check(hbDNNInferV2(&task,session.outputs.data(),session.inputs.data(),session.model),"hbDNNInferV2");
  hbUCPSchedParam sched;HB_UCP_INITIALIZE_SCHED_PARAM(&sched);submission_attempted=true;
  check(hbUCPSubmitTask(task,&sched),"hbUCPSubmitTask");
  int rc=hbUCPWaitTaskDone(task,int32_t(timeout));
  require(rc==0,rc==HB_UCP_TASK_TIMEOUT?VITA_TIMEOUT:VITA_RUNTIME,"UCP completion not established, buffers retained; wait rc="+std::to_string(rc));
  check(hbUCPReleaseTask(task),"hbUCPReleaseTask");task=nullptr;submission_attempted=false;
 }
 bool drain(uint32_t timeout){
  if(!task)return true;
  // Never destroy the graph/buffers while submitted device work may still own them.
  if(submission_attempted&&hbUCPWaitTaskDone(task,int32_t(timeout))!=0)return false;
  if(hbUCPReleaseTask(task)!=0)return false;
  task=nullptr;submission_attempted=false;return true;
 }
};
struct Backend {
 std::mutex api;
 Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"vita_shadow_backend"};
 uint32_t timeout;bool parallel;double deadline;bool poisoned=false;
 uint64_t velocity_calls=0,motion_calls=0,explicit_resets=0,rope_resets=0;
 std::unique_ptr<policy_cut::Frontend> front;
 std::unique_ptr<policy_cut::Backend> back;
 std::unique_ptr<VelocityOrt> velocity;
 std::unique_ptr<vita_router::Router> router;
 std::vector<float> pool0,pool1;
 UcpRun moe,velocity_hbm;
 size_t wi0=0,wi1=0,gi=0,hi=0,mo=0,vi=0,vo=0;
 vita_ucp::Layout wl0,wl1,gl,hl,ml,vl_in,vl_out;
 bool has_velocity_hbm=false;
 // Destroyed first; referenced observations/router outlive worker.
 std::unique_ptr<vita_router::Worker> worker;
 explicit Backend(const VitaConfig& cfg):timeout(cfg.timeout_ms),parallel(cfg.parallel_router!=0),deadline(cfg.deadline_ms){
  require(timeout>0&&timeout<=60000&&cfg.parallel_router<=1&&std::isfinite(deadline)&&deadline>0,VITA_INVALID,"Invalid timeout/parallel/deadline");
  for(const auto* a:{&cfg.front,&cfg.back,&cfg.moe,&cfg.router,&cfg.velocity})verify(*a);
  front=std::make_unique<policy_cut::Frontend>(env,cfg.front.path);back=std::make_unique<policy_cut::Backend>(env,cfg.back.path);velocity=std::make_unique<VelocityOrt>(env,cfg.velocity.path);router=std::make_unique<vita_router::Router>(cfg.router.path);
  pool0=pool(cfg.pool0,512*512);pool1=pool(cfg.pool1,512*256);
  auto& s=moe.session;s.load(cfg.moe.path);require(s.inputs.size()==4&&s.outputs.size()==1,VITA_INVALID,"MoE HBM must have four inputs/one output");
  wi0=vita_ucp::index(s.inames,"selected_expert_weights_0");wi1=vita_ucp::index(s.inames,"selected_expert_weights_1");gi=vita_ucp::index(s.inames,"normalized_expert_gates");hi=vita_ucp::index(s.inames,"norm2_hidden");mo=vita_ucp::index(s.onames,"moe_update");
  wl0=vita_ucp::layout(s.inputs[wi0].properties,{2,512,512},HB_DNN_TENSOR_TYPE_F32,4);wl1=vita_ucp::layout(s.inputs[wi1].properties,{2,512,256},HB_DNN_TENSOR_TYPE_F32,4);gl=vita_ucp::layout(s.inputs[gi].properties,{1,2,1},HB_DNN_TENSOR_TYPE_F32,4);hl=vita_ucp::layout(s.inputs[hi].properties,{1,1,512},HB_DNN_TENSOR_TYPE_F32,4);ml=vita_ucp::layout(s.outputs[mo].properties,{1,1,512},HB_DNN_TENSOR_TYPE_F32,4);
  if(cfg.velocity_hbm.path&&*cfg.velocity_hbm.path){verify(cfg.velocity_hbm);auto& v=velocity_hbm.session;v.load(cfg.velocity_hbm.path);require(v.inputs.size()==1&&v.outputs.size()==1,VITA_INVALID,"Velocity HBM must have one input/output");vi=vita_ucp::index(v.inames,"obs");vo=vita_ucp::index(v.onames,"actions");vl_in=vita_ucp::layout(v.inputs[vi].properties,{1,990},HB_DNN_TENSOR_TYPE_F32,4);vl_out=vita_ucp::layout(v.outputs[vo].properties,{1,28},HB_DNN_TENSOR_TYPE_F32,4);has_velocity_hbm=true;}
  if(parallel)worker=std::make_unique<vita_router::Worker>(*router);
 }
 void healthy(){require(!poisoned,VITA_POISONED,"Session poisoned by previous execution failure; close/recreate after task drains");}
 void reset_state(bool automatic){std::fill(front->past_cache.begin(),front->past_cache.end(),0.f);std::fill(front->present_cache.begin(),front->present_cache.end(),0.f);front->step=0;if(automatic)++rope_resets;else ++explicit_resets;}
 void init_timing(VitaTiming& t){std::memset(&t,0,sizeof(t));t.abi_version=VITA_BACKEND_ABI;t.struct_bytes=sizeof(t);t.step_used=-1;t.step_next=front->step;}
 void finish_timing(VitaTiming& t,Clock::time_point begin){t.velocity_calls=velocity_calls;t.motion_calls=motion_calls;t.explicit_resets=explicit_resets;t.rope_resets=rope_resets;t.step_next=front->step;t.total_ms=elapsed(begin,Clock::now());t.deadline_missed=t.total_ms>deadline;}
 void run_velocity(const float* obs,int mode,float* actions,VitaTiming& t){
  healthy();require(mode==VITA_VELOCITY_ORT||(mode==VITA_VELOCITY_HBM&&has_velocity_hbm),VITA_INVALID,"Velocity backend unavailable");auto begin=Clock::now();finite(obs,990,"Velocity input");init_timing(t);
  try{
   if(mode==VITA_VELOCITY_ORT){std::copy(obs,obs+990,velocity->observation.begin());auto start=Clock::now();velocity->run();t.backend_ms=elapsed(start,Clock::now());}
   else{auto& s=velocity_hbm.session;vita_ucp::pack(s.inputs[vi],vl_in,obs);check(hbUCPMemFlush(&s.inputs[vi].sysMem,HB_SYS_MEM_CACHE_CLEAN),"Velocity CLEAN");auto start=Clock::now();t.prepare_ms=elapsed(begin,start);velocity_hbm.execute(timeout);auto end=Clock::now();t.moe_ms=elapsed(start,end);check(hbUCPMemFlush(&s.outputs[vo].sysMem,HB_SYS_MEM_CACHE_INVALIDATE),"Velocity INVALIDATE");vita_ucp::unpack(s.outputs[vo],vl_out,velocity->actions.data());t.backend_ms=elapsed(end,Clock::now());}
   auto out=Clock::now();finite(velocity->actions.data(),28,"Velocity output");std::copy(velocity->actions.begin(),velocity->actions.end(),actions);++velocity_calls;t.output_ms=elapsed(out,Clock::now());finish_timing(t,begin);
  }catch(...){poisoned=true;throw;}
 }
 void run_motion(const float* obs,float* actions,float* cache,int64_t* routes,float* logits,VitaTiming& t){
  healthy();auto begin=Clock::now();finite(obs,3126,"Motion input");init_timing(t);
  try{
   if(front->step>=8128){reset_state(true);t.automatic_reset=1;}t.step_used=front->step;
   std::copy(obs,obs+3126,front->observation.begin());auto parallel_start=Clock::now();
   if(parallel)worker->submit(front->observation.data());else{auto r=Clock::now();router->run(front->observation.data());t.router_ms=elapsed(r,Clock::now());}
   auto f=Clock::now();front->run();t.frontend_ms=elapsed(f,Clock::now());
   if(parallel)require(worker->wait(timeout,t.router_ms),VITA_TIMEOUT,"Router worker timeout; session retained");
   auto p=Clock::now();t.parallel_ms=elapsed(parallel_start,p);
   finite(front->hidden.data(),512,"Norm2");finite(front->residual.data(),512,"Attention residual");finite(front->present_cache.data(),16384,"KV");finite(router->logits.data(),1024,"Router logits");finite(router->gates.data(),2,"Router gates");
   auto& s=moe.session;vita_ucp::pack_selected_rows(s.inputs[wi0],wl0,pool0,512*512,router->ids);vita_ucp::pack_selected_rows(s.inputs[wi1],wl1,pool1,512*256,router->ids);vita_ucp::pack(s.inputs[gi],gl,router->gates.data());vita_ucp::pack(s.inputs[hi],hl,front->hidden.data());for(auto& input:s.inputs)check(hbUCPMemFlush(&input.sysMem,HB_SYS_MEM_CACHE_CLEAN),"MoE input CLEAN");
   auto g=Clock::now();t.prepare_ms=elapsed(p,g);moe.execute(timeout);auto b=Clock::now();t.moe_ms=elapsed(g,b);
   check(hbUCPMemFlush(&s.outputs[mo].sysMem,HB_SYS_MEM_CACHE_INVALIDATE),"MoE output INVALIDATE");vita_ucp::unpack(s.outputs[mo],ml,back->moe_update.data());finite(back->moe_update.data(),512,"MoE update");std::copy(front->residual.begin(),front->residual.end(),back->residual.begin());back->run();auto out=Clock::now();t.backend_ms=elapsed(b,out);
   finite(back->actions.data(),30,"Motion output");
   // Commit recurrence only once the complete policy succeeded and all outputs are finite.
   front->feedback();++front->step;++motion_calls;
   std::copy(back->actions.begin(),back->actions.end(),actions);
   if(cache)std::copy(front->present_cache.begin(),front->present_cache.end(),cache);
   if(routes){routes[0]=router->ids[0];routes[1]=router->ids[1];}
   if(logits)std::copy(router->logits.begin(),router->logits.end(),logits);
   t.output_ms=elapsed(out,Clock::now());finish_timing(t,begin);
  }catch(...){poisoned=true;throw;}
 }
 bool ready_to_destroy(){return (!worker||worker->idle())&&moe.drain(timeout)&&velocity_hbm.drain(timeout);}
};
Backend& get(void* p){require(p!=nullptr,VITA_INVALID,"Null backend");return *static_cast<Backend*>(p);}
std::unique_lock<std::mutex> lock(Backend& b){std::unique_lock<std::mutex> l(b.api,std::try_to_lock);require(l.owns_lock(),VITA_BUSY,"Concurrent backend call rejected");return l;}
void timing_ptr(VitaTiming* t){require(t!=nullptr&&t->abi_version==VITA_BACKEND_ABI&&t->struct_bytes==sizeof(*t),VITA_INVALID,"Invalid timing ABI/buffer");}
void array(const void* p,size_t n,size_t expected,const char* label){require(p&&n==expected,VITA_INVALID,std::string("Invalid buffer size: ")+label);}
void optional_array(const void* p,size_t n,size_t expected,const char* label){require((!p&&n==0)||(p&&n==expected),VITA_INVALID,std::string("Invalid optional buffer: ")+label);}
}
extern "C" {
const char* vita_build_info(){return "vita_backend_abi1; original_fp32; no_actuation; ORT+UCP; raw_actions; motion3126x30; velocity990x28";}
int vita_create(const VitaConfig* c,void** h,char* e,size_t n){if(h)*h=nullptr;return guard(e,n,[&]{require(c&&h&&c->abi_version==VITA_BACKEND_ABI&&c->struct_bytes==sizeof(*c),VITA_INVALID,"Invalid config ABI");auto b=std::make_unique<Backend>(*c);*h=b.release();});}
int vita_reset(void* h,char* e,size_t n){return guard(e,n,[&]{auto& b=get(h);auto l=lock(b);b.healthy();require(!b.worker||b.worker->idle(),VITA_BUSY,"Router still running");b.reset_state(false);});}
int vita_state(void* h,VitaState* s,char* e,size_t n){return guard(e,n,[&]{auto& b=get(h);auto l=lock(b);require(s&&s->abi_version==VITA_BACKEND_ABI&&s->struct_bytes==sizeof(*s),VITA_INVALID,"Invalid state ABI");s->velocity_calls=b.velocity_calls;s->motion_calls=b.motion_calls;s->explicit_resets=b.explicit_resets;s->rope_resets=b.rope_resets;s->motion_step=b.front->step;s->poisoned=b.poisoned;s->velocity_hbm_available=b.has_velocity_hbm;});}
int vita_velocity(void* h,const float* obs,size_t no,int mode,float* a,size_t na,VitaTiming* t,char* e,size_t n){return guard(e,n,[&]{auto& b=get(h);auto l=lock(b);array(obs,no,990,"obs");array(a,na,28,"actions");timing_ptr(t);b.run_velocity(obs,mode,a,*t);});}
int vita_motion(void* h,const float* obs,size_t no,float* a,size_t na,float* c,size_t nc,int64_t* r,size_t nr,float* logits,size_t nl,VitaTiming* t,char* e,size_t n){return guard(e,n,[&]{auto& b=get(h);auto l=lock(b);array(obs,no,3126,"obs");array(a,na,30,"actions");optional_array(c,nc,16384,"cache");optional_array(r,nr,2,"routes");optional_array(logits,nl,1024,"logits");timing_ptr(t);b.run_motion(obs,a,c,r,logits,*t);});}
int vita_destroy(void* h,char* e,size_t n){if(!h)return VITA_OK;return guard(e,n,[&]{auto& b=get(h);{auto l=lock(b);require(b.ready_to_destroy(),VITA_TIMEOUT,"Outstanding worker/device work; handle and all buffers retained, retry close later");}delete &b;});}
}
