// ASan/UBSan smoke for bounded traversal; argv: captured length-prefixed file.
#include "../src/deployment/vita_s100_aorta_30dof/onboard_state_native.cpp"
#include <fstream>
#include <vector>
#include <random>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ifstream file(argv[1],std::ios::binary);uint32_t size=0;file.read(reinterpret_cast<char*>(&size),4);
 if(size<4||size>1024*1024)return 3;
 std::vector<uint8_t> original(size);file.read(reinterpret_cast<char*>(original.data()),size);if(!file)return 4;
 int32_t slots[30],roles[44];for(int j=0;j<30;++j)slots[j]=j;for(int j=0;j<44;++j)roles[j]=j<42?1:2;
 float out[67];uint64_t meta[2];char error[192];std::mt19937 rng(314159);
 for(int run=0;run<100000;++run){auto buf=original;auto n=run%4==0?rng()%(size+1):size;
  for(unsigned k=0;k<1+rng()%8;++k)buf[rng()%size]=uint8_t(rng());
  vita_state_decode(buf.data(),n,slots,roles,out,meta,error,sizeof(error));
 }
 std::cout<<"PASS: 100000 seeded mutations/truncations under ASan+UBSan\n";
}
