#pragma once
// Vita model_20000 exact FP32 frontend/backend cuts. ABI checked on load.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace policy_cut {
inline void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
struct Port {const char* name;std::vector<int64_t> shape;ONNXTensorElementDataType type;};
inline Ort::SessionOptions options(int threads=1){
 Ort::SessionOptions result;result.SetIntraOpNumThreads(threads);result.SetInterOpNumThreads(1);
 result.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);result.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
 // Sleep idle workers between 50Hz calls instead of competing with callbacks.
 result.AddConfigEntry("session.intra_op.allow_spinning", "0");
 return result;
}
inline void validate(Ort::Session& session,const std::vector<Port>& expected,bool output){
 size_t count=output?session.GetOutputCount():session.GetInputCount();require(count==expected.size(),"Unexpected subgraph IO count");
 Ort::AllocatorWithDefaultOptions allocator;
 for(size_t i=0;i<count;++i){
  auto name=output?session.GetOutputNameAllocated(i,allocator):session.GetInputNameAllocated(i,allocator);
  auto found=std::find_if(expected.begin(),expected.end(),[&](const Port& p){return p.name==std::string(name.get());});
  require(found!=expected.end(),std::string("Unexpected port: ")+name.get());
  auto type=output?session.GetOutputTypeInfo(i):session.GetInputTypeInfo(i);
  require(type.GetONNXType()==ONNX_TYPE_TENSOR,"Expected tensor port");auto info=type.GetTensorTypeAndShapeInfo();
  require(info.GetElementType()==found->type && info.GetShape()==found->shape,std::string("Unexpected dtype/shape: ")+name.get());
 }
}
template<class T>inline Ort::Value tensor(T* data,size_t count,const std::vector<int64_t>& shape){
 auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
 return Ort::Value::CreateTensor<T>(memory,data,count,shape.data(),shape.size());
}
struct Frontend {
 Ort::Session session;
 std::vector<float> observation=std::vector<float>(3126),past_cache=std::vector<float>(16384,0),hidden=std::vector<float>(512),residual=std::vector<float>(512),present_cache=std::vector<float>(16384);
 int64_t step=0;
 std::array<const char*,3> input_names{{"obs","past_key_values","step_idx"}};
 std::array<const char*,3> output_names{{"/actor_module/norm2/Mul_1_output_0","/actor_module/Add_4_output_0","present_key_values"}};
 std::vector<Ort::Value> inputs,outputs;
 explicit Frontend(Ort::Env& env,const char* path):session(env,path,options(3)){
  validate(session,{{input_names[0],{1,3126},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},{input_names[1],{1,2,1,32,4,64},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},{input_names[2],{1},ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}},false);
  validate(session,{{output_names[0],{1,1,512},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},{output_names[1],{1,1,512},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},{output_names[2],{1,2,1,32,4,64},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}},true);
  inputs.emplace_back(tensor(observation.data(),observation.size(),{1,3126}));inputs.emplace_back(tensor(past_cache.data(),past_cache.size(),{1,2,1,32,4,64}));inputs.emplace_back(tensor(&step,1,{1}));
  outputs.emplace_back(tensor(hidden.data(),hidden.size(),{1,1,512}));outputs.emplace_back(tensor(residual.data(),residual.size(),{1,1,512}));outputs.emplace_back(tensor(present_cache.data(),present_cache.size(),{1,2,1,32,4,64}));
 }
 Frontend(const Frontend&)=delete;Frontend& operator=(const Frontend&)=delete;
 void run(){require(step>=0 && step<8192,"Original absolute RoPE index out of bounds");session.Run(Ort::RunOptions{nullptr},input_names.data(),inputs.data(),inputs.size(),output_names.data(),outputs.data(),outputs.size());}
 void feedback(){std::copy(present_cache.begin(),present_cache.end(),past_cache.begin());}
};
struct Backend {
 Ort::Session session;
 std::vector<float> residual=std::vector<float>(512),moe_update=std::vector<float>(512),actions=std::vector<float>(30);
 std::array<const char*,2> input_names{{"/actor_module/Add_4_output_0","/actor_module/Add_6_output_0"}};
 std::array<const char*,1> output_names{{"actions"}};
 std::vector<Ort::Value> inputs,outputs;
 explicit Backend(Ort::Env& env,const char* path):session(env,path,options()){
  validate(session,{{input_names[0],{1,1,512},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},{input_names[1],{1,1,512},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}},false);
  validate(session,{{output_names[0],{1,30},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}},true);
  inputs.emplace_back(tensor(residual.data(),residual.size(),{1,1,512}));inputs.emplace_back(tensor(moe_update.data(),moe_update.size(),{1,1,512}));outputs.emplace_back(tensor(actions.data(),actions.size(),{1,30}));
 }
 Backend(const Backend&)=delete;Backend& operator=(const Backend&)=delete;
 void run(){session.Run(Ort::RunOptions{nullptr},input_names.data(),inputs.data(),inputs.size(),output_names.data(),outputs.data(),outputs.size());}
};
} // namespace policy_cut
