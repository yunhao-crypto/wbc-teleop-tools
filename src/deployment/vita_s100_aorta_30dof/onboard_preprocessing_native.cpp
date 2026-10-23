// Portable FP32 implementation of the released Warp preprocessing expressions.
// No Warp/Torch/CUDA runtime; called from the validated NumPy state wrapper.
#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef PP_DOUBLE_TRIG
static float psin(float x){return float(std::sin(double(x)));}
static float pcos(float x){return float(std::cos(double(x)));}
static float patan2(float y,float x){return float(std::atan2(double(y),double(x)));}
#else
static float psin(float x){return std::sin(x);}
static float pcos(float x){return std::cos(x);}
static float patan2(float y,float x){
 // Match FP32 quadrant reduction used by original GPU math. The reduced
 // angle uses the platform libm; no lookup table or trajectory correction.
 float ay=std::abs(y),ax=std::abs(x);
 if(!std::isfinite(ay+ax)||std::max(ay,ax)==0.f)return std::atan2(y,x);
 float a=std::atan(std::min(ay,ax)/std::max(ay,ax));
 if(ay>ax)a=1.5707963705062866211f-a;
 if(x<0.f)a=3.1415927410125732422f-a;
 return std::copysign(a,y);
}
#endif
struct Q {float w,x,y,z;};
struct V {float x,y,z;};
static Q qm(Q a,Q b){return {a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
 a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
 a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
 a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w};}
static Q qc(Q q){return {q.w,-q.x,-q.y,-q.z};}
static float qdot(Q a,Q b){float v=0;v+=a.w*b.w;v+=a.x*b.x;v+=a.y*b.y;v+=a.z*b.z;return v;}
static Q qn(Q q,bool reciprocal=false,bool positive=false){
 float norm2=std::fma(q.z,q.z,std::fma(q.y,q.y,std::fma(q.w,q.w,q.x*q.x)));
 float n=std::sqrt(norm2);n=std::max(n,reciprocal?1e-8f:1e-9f);
 if(reciprocal){float s=1.f/n;q={q.w*s,q.x*s,q.y*s,q.z*s};}
 else q={q.w/n,q.x/n,q.y/n,q.z/n};
 if(positive&&q.w<0)q={-q.w,-q.x,-q.y,-q.z};return q;
}
static Q qload(const float* p){return {p[0],p[1],p[2],p[3]};}
static void qstore(float* p,Q q){p[0]=q.w;p[1]=q.x;p[2]=q.y;p[3]=q.z;}
static V vload(const float* p){return {p[0],p[1],p[2]};}
static void vstore(float* p,V v){p[0]=v.x;p[1]=v.y;p[2]=v.z;}
static V rot(Q q,V v){
 float tx=std::fma(q.y,v.z,-(q.z*v.y));tx=tx+tx;
 float ty=std::fma(q.z,v.x,-(q.x*v.z));ty=ty+ty;
 float tz=std::fma(q.x,v.y,-(q.y*v.x));tz=tz+tz;
 return {std::fma(q.w,tx,v.x)+std::fma(q.y,tz,-(q.z*ty)),
         std::fma(q.w,ty,v.y)+std::fma(q.z,tx,-(q.x*tz)),
         std::fma(q.w,tz,v.z)+std::fma(q.x,ty,-(q.y*tx))};
}
static V warp_inv(Q q,V v){
 // Explicit contractions match released Warp sm86 SASS (pack kernel).
 float c=std::fma(q.w,q.w+q.w,-1.f);
 float d=std::fma(q.z,v.z,std::fma(q.x,v.x,q.y*v.y));d=d+d;
 float cx=std::fma(q.y,v.z,-(q.z*v.y));
 float cy=std::fma(q.z,v.x,-(q.x*v.z));
 float cz=std::fma(q.x,v.y,-(q.y*v.x));
 return {std::fma(c,v.x,q.x*d)-std::fma(q.w,cx,q.w*cx),
         std::fma(c,v.y,q.y*d)-std::fma(q.w,cy,q.w*cy),
         std::fma(c,v.z,q.z*d)-std::fma(q.w,cz,q.w*cz)};
}
static Q aa(float x,float y,float z){float t=std::fma(z,z,std::fma(x,x,y*y));if(t<1e-8f)return {1,.5f*x,.5f*y,.5f*z};
 t=std::sqrt(t);float half=.5f*t,s=psin(half)/t;return {pcos(half),x*s,y*s,z*s};}
static float yaw(Q q){return patan2(2.f*(q.w*q.z+q.x*q.y),1.f-2.f*(q.y*q.y+q.z*q.z));}
static void r6(float* p,Q q){p[0]=1.f-2.f*(q.y*q.y+q.z*q.z);p[1]=2.f*(q.x*q.y-q.z*q.w);p[2]=2.f*(q.x*q.y+q.z*q.w);p[3]=1.f-2.f*(q.x*q.x+q.z*q.z);p[4]=2.f*(q.x*q.z-q.y*q.w);p[5]=2.f*(q.y*q.z+q.x*q.w);}
extern "C" void pp_pico(const float* body,const int32_t* parents,float* pose,float* trans){
 Q globals[24];for(int j=0;j<24;++j){const float* b=body+j*7;Q q=qn({b[6],b[3],b[4],b[5]},true);float s=.7071067811865476f;
 globals[j]=qn(qm(qm({s,s,0,0},q),{0,0,1,0}),true);}
 trans[0]=body[0];trans[1]=-body[2];trans[2]=body[1];
 for(int j=0;j<24;++j){Q q=globals[j];if(parents[j]>=0)q=qn(qm(qc(globals[parents[j]]),q),true);
 if(q.w<0)q={-q.w,-q.x,-q.y,-q.z};float mag=std::sqrt(std::fma(q.z,q.z,std::fma(q.x,q.x,q.y*q.y))),s=2.f;
 if(mag>1e-7f)s=2.f*patan2(mag,std::max(q.w,0.f))/mag;
 pose[j*3]=q.x*s;pose[j*3+1]=q.y*s;pose[j*3+2]=q.z*s;}
}
extern "C" void pp_features(const float* pose,const float* trans,const int32_t* parents,const float* offsets,float* body,float* foot,float* root){
 Q qs[24];V pos[24];pos[0]={0,0,0};
 for(int j=0;j<24;++j){Q local=aa(pose[j*3],pose[j*3+1],pose[j*3+2]);if(j==0){qs[0]=local;continue;}
 int p=parents[j];qs[j]=qn(qm(qs[p],local),true);V v=rot(qs[p],vload(offsets+3*j));pos[j]={pos[p].x+v.x,pos[p].y+v.y,pos[p].z+v.z};}
 for(int j=1;j<24;++j){V native=warp_inv(qs[0],pos[j]);body[(j-1)*3]=native.z;body[(j-1)*3+1]=native.x;body[(j-1)*3+2]=native.y;}
 for(int i=69;i<132;++i)body[i]=pose[i-66];
 Q command{.5f,-.5f,-.5f,-.5f};
 for(int j=1;j<=21;++j){Q rel=qm(qc(qs[0]),qs[j]);Q semantic=qn(qm(qm(qc(command),rel),command),false,true);r6(body+132+(j-1)*6,semantic);}
 *foot=trans[2]+std::min(pos[10].z,pos[11].z);qstore(root,qn(qm(qs[0],command),false,true));
}
extern "C" void pp_gather(const float* bodies,const float* feet,const float* trans,const float* roots,const float* times,int oldest,int capacity,int ground_frames,int seq_frames,int future,int existing,float* anchor,float* out,float* sample,float* bodyout){
 int start=ground_frames-1;int a=(oldest+start+1)%capacity;
 if(!existing){anchor[0]=trans[a*3];anchor[1]=trans[a*3+1];anchor[2]=yaw(qload(roots+a*4));}
 float ny=-anchor[2];Q yq{pcos(.5f*ny),0,0,psin(.5f*ny)};float cy=pcos(ny),sy=psin(ny);
 float first=times[(oldest+start)%capacity];Q previous{1,0,0,0};
 for(int f=0;f<seq_frames;++f){int c=start+f,s=(oldest+c)%capacity;float ground=1e30f;
 for(int h=0;h<ground_frames;++h)if(c-h>=0)ground=std::min(ground,feet[(oldest+c-h)%capacity]);
 float x=trans[s*3]-anchor[0],y=trans[s*3+1]-anchor[1];out[f*7]=cy*x-sy*y;out[f*7+1]=sy*x+cy*y;out[f*7+2]=trans[s*3+2]-ground;
 Q q=qn(qm(yq,qload(roots+s*4)),false,true);if(f>0&&qdot(previous,q)<0)q={-q.w,-q.x,-q.y,-q.z};qstore(out+f*7+3,q);previous=q;sample[f]=times[s]-first;}
 for(int f=0;f<=future;++f){int s=(oldest+start+1+f)%capacity;std::copy(bodies+s*258,bodies+(s+1)*258,bodyout+f*258);}
}
static V angular(Q q0,Q q1,float dt){Q r{q1.w*q0.w+q1.x*q0.x+q1.y*q0.y+q1.z*q0.z,
 -q1.w*q0.x+q1.x*q0.w-q1.y*q0.z+q1.z*q0.y,
 -q1.w*q0.y+q1.x*q0.z+q1.y*q0.w-q1.z*q0.x,
 -q1.w*q0.z-q1.x*q0.y+q1.y*q0.x+q1.z*q0.w};
 if(r.w<0)r={-r.w,-r.x,-r.y,-r.z};float inv=1.f/std::max(std::sqrt(r.w*r.w+r.x*r.x+r.y*r.y+r.z*r.z),1e-9f);r={r.w*inv,r.x*inv,r.y*inv,r.z*inv};
 float mag=std::sqrt(r.x*r.x+r.y*r.y+r.z*r.z),s=2.f;if(mag>1e-7f)s=2.f*patan2(mag,r.w)/mag;return {r.x*s/dt,r.y*s/dt,r.z*s/dt};}
extern "C" void pp_build(const float* root,const float* times,const float* body,const float* rq0,const float* av,const float* pos,const float* vel,const float* last,const float* defaults,const float* align0,int future,int dofs,float* out){
 // Current and future kinematic terms exclude the two derivative-only boundaries.
 Q rq=qn(qload(rq0)),align=qn(qload(align0));int curdim=258+18+3*dofs;
 for(int f=1;f<=future+1;++f){int lo=f-1,hi=f+1;float dt=std::max(times[hi]-times[lo],1e-6f);Q q=qload(root+f*7+3);
 V lw{(root[hi*7]-root[lo*7])/dt,(root[hi*7+1]-root[lo*7+1])/dt,(root[hi*7+2]-root[lo*7+2])/dt};
 V aw=angular(qload(root+lo*7+3),qload(root+hi*7+3),dt),ll=rot(qc(q),lw),al=rot(qc(q),aw),g=rot(qc(q),{0,0,-1});
 Q ref=qn(q),aligned=qn(qm(align,ref));
 if(f==1){vstore(out,g);vstore(out+3,ll);vstore(out+6,al);std::copy(body,body+258,out+9);out[267]=root[f*7+2];float ye=yaw(aligned)-yaw(rq);out[268]=psin(ye);out[269]=pcos(ye);
 out[270]=2.f*(-rq.z*rq.x+rq.w*rq.y);out[271]=-2.f*(rq.z*rq.y+rq.w*rq.x);out[272]=1.f-2.f*(rq.w*rq.w+rq.z*rq.z);
 std::copy(av,av+3,out+273);for(int d=0;d<dofs;++d){out[276+d]=pos[d]-defaults[d];out[276+dofs+d]=vel[d];out[276+2*dofs+d]=last[d];}}
 else{int j=f-2,b=curdim;std::copy(body+(j+1)*258,body+(j+2)*258,out+b+j*258);b+=future*258;out[b+j]=root[f*7+2];b+=future;vstore(out+b+j*3,g);b+=future*3;vstore(out+b+j*3,ll);b+=future*3;vstore(out+b+j*3,al);b+=future*3;
 float yd=yaw(ref)-yaw(qn(qload(root+10)));out[b+j*2]=psin(yd);out[b+j*2+1]=pcos(yd);b+=future*2;r6(out+b+j*6,qn(qm(qc(rq),aligned)));}
 }
}
// Diagnostic-only exports: do not change the production expression paths.
extern "C" void pp_probe_fk(const float* pose,const int32_t* parents,const float* offsets,float* posout,float* quatout){
 Q qs[24];V pos[24];pos[0]={0,0,0};
 for(int j=0;j<24;++j){Q local=aa(pose[j*3],pose[j*3+1],pose[j*3+2]);if(j==0){qs[0]=local;continue;}
 int p=parents[j];qs[j]=qn(qm(qs[p],local),true);V v=rot(qs[p],vload(offsets+3*j));pos[j]={pos[p].x+v.x,pos[p].y+v.y,pos[p].z+v.z};}
 for(int j=0;j<24;++j){vstore(posout+j*3,pos[j]);qstore(quatout+j*4,qs[j]);}
}
extern "C" void pp_probe_roundtrip(const float* root,const float* offset,float* pos,float* local){Q q=qload(root);V v=rot(q,vload(offset));vstore(pos,v);vstore(local,warp_inv(q,v));}

extern "C" void pp_probe_pico_global(const float* body,float* out){for(int j=0;j<24;++j){const float* b=body+j*7;Q q=qn({b[6],b[3],b[4],b[5]},true);float s=.7071067811865476f;qstore(out+j*4,qn(qm(qm({s,s,0,0},q),{0,0,1,0}),true));}}
