#!/usr/bin/env python3
"""Compile actual callback/getter into an SDK-free concurrent regression test."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
binding=root/'third_party/pico_binding'
src=(binding/'bindings/py_bindings.cpp').read_text()
# Compile the production globals/parser/callback and getter verbatim. Exclude
# Python module registration and SDK init/control functions; no test API export.
head=src[:src.index('void init()')]
head='\n'.join(line for line in head.splitlines() if not line.startswith('#include <pybind11/'))
getter=src[src.index('std::tuple<bool, int64_t, std::array<std::array<double, 7>, 24>> getBodyPoseSnapshot()'):src.index('bool isBodyDataAvailable()')]
getter+=src[src.index('std::tuple<bool, int64_t, double, double, double> getControllerTriggerSnapshot()'):src.index('double getLeftGrip()')]
test=r'''
#include <atomic>
#include <cassert>
#include <cstring>
json packet(int64_t frame) {
 json joints=json::array();
 std::string encoded;
 for(int k=0;k<7;++k){if(k)encoded+=",";encoded+=std::to_string(frame);}
 for(int j=0;j<24;++j)joints.push_back({{"p",encoded}});
 return {{"Controller",json::object()},{"Hand",json::object()},
         {"timeStampNs",frame},{"Body",{{"timeStampNs",frame+1000000},{"joints",joints}}}};
}
void publish(const json& value) {
 const std::string encoded=json{{"value",value.dump()}}.dump();
 PXREADevStateJson state{};assert(encoded.size()<sizeof(state.stateJson));
 std::memcpy(state.stateJson,encoded.c_str(),encoded.size()+1);
 OnPXREAClientCallback(nullptr,PXREADeviceStateJson,0,&state);
}
void coherent(){auto [available,stamp,pose]=getBodyPoseSnapshot();if(!available)return;
 for(const auto&joint:pose)for(double coordinate:joint)assert(coordinate==double(stamp));}
json with_controller(json value) {
 const double depth=double(value["timeStampNs"].get<int64_t>()%2);
 const json side={{"pose","0,0,0,0,0,0,1"},{"trigger",depth},{"grip",0.},
   {"menuButton",false},{"axisX",0.},{"axisY",0.},{"axisClick",false},
   {"primaryButton",false},{"secondaryButton",false}};
 value["Controller"]={{"left",side},{"right",side}};return value;
}
void triggers_coherent(){auto [available,stamp,age,left,right]=getControllerTriggerSnapshot();
 if(!available)return;assert(age>=0);assert(left==right);assert(left==double(stamp%2));}
int main(){
 assert(!std::get<0>(getControllerTriggerSnapshot()));
 publish(with_controller(packet(1)));triggers_coherent();
 auto bad_controller=with_controller(packet(2));bad_controller["Controller"].erase("right");
 publish(bad_controller);assert(!std::get<0>(getControllerTriggerSnapshot()));
 bad_controller=with_controller(packet(3));bad_controller["Controller"]["left"]["trigger"]=1.2;
 publish(bad_controller);assert(!std::get<0>(getControllerTriggerSnapshot()));
 {std::lock_guard<std::mutex>lock(bodyMutex);AtomicBodyPose=BodyPoseSnapshot{};}
 assert(!std::get<0>(getBodyPoseSnapshot()));publish(packet(10));coherent();
 assert(std::get<1>(getBodyPoseSnapshot())==10); // root, not Body timestamp
 {std::lock_guard<std::mutex>lock(timestampMutex);TimeStampNs=11;}
 coherent();assert(std::get<1>(getBodyPoseSnapshot())==10); // global update cannot tear body
 auto invalid=packet(11);invalid["Body"]["joints"].erase(23);publish(invalid);
 assert(std::get<1>(getBodyPoseSnapshot())==10);coherent();
 invalid=packet(12);invalid["Body"]["joints"][5].erase("p");publish(invalid);
 assert(std::get<1>(getBodyPoseSnapshot())==10);
 for(auto bad:{"1,2,3","1,2,3,4,5,6,7,8","1,2,3,4,5,6,nan","1,2,3,4,5,6,7,"}){
  invalid=packet(13);invalid["Body"]["joints"][20]["p"]=bad;publish(invalid);
  assert(std::get<1>(getBodyPoseSnapshot())==10);coherent();
 }
 invalid=packet(14);invalid.erase("timeStampNs");publish(invalid);
 assert(std::get<1>(getBodyPoseSnapshot())==10);
 std::atomic<bool>done{false};std::atomic<unsigned>reads{0};
 std::thread reader([&]{while(!done){coherent();triggers_coherent();++reads;}});
 for(int frame=20;frame<10020;++frame)publish(with_controller(packet(frame)));
 done=true;reader.join();coherent();assert(reads>0);
 std::cout<<"PASS: actual callback/getter, root timestamp semantics, torn global update, 7 malformed packets, 10000 concurrent publications, "<<reads<<" coherent reads\n";
}
'''
out=root/'build/state';out.mkdir(parents=True,exist_ok=True)
source=out/'pico_snapshot_test.cpp';source.write_text(head+getter+test)
binary=out/'pico_snapshot_test'
subprocess.run(['c++','-std=c++17','-O2','-pthread','-I'+str(binding/'include'),'-I'+str(binding/'bindings'),str(source),'-o',str(binary)],check=True)
subprocess.run([str(binary)],check=True)
