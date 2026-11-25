#include "config.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef CPU_ONLY
#include "acl/acl.h"
extern "C" void launch_int8_tiles(void*,void*,void*,uint32_t,aclrtStream);
#endif
using Wide=__int128_t;
using UWide=__uint128_t;
using Digits=std::array<int8_t,5>;
static unsigned planes(bool e5){return e5?5:3;}
static unsigned q(bool e5){return e5?16:9;}
static bool finite(uint8_t raw,bool e5){return e5?((raw&0x7c)!=0x7c):((raw&0x7f)!=0x7f);}
static Digits decode(uint8_t raw,bool e5){
    if(!finite(raw,e5)) throw std::runtime_error("Nonfinite FP8 rejected");
    const unsigned mb=e5?2:3, e=(raw>>mb)&(e5?31:15), m=raw&((1u<<mb)-1);
    uint32_t mag=e?((1u<<mb)+m)<<(e-1):m;
    Digits d{};
    for(unsigned i=0;i<planes(e5);++i){int v=mag&127u;d[i]=int8_t(raw&128?-v:v);mag>>=7;}
    if(mag) throw std::runtime_error("Digit decomposition overflow");
    return d;
}
// Independent reference uses mathematical FP8 definition, not digit splitting.
// Each individual value and scaled integer is exactly representable in double.
static int64_t reference(uint8_t raw,bool e5){
    if(!finite(raw,e5)) throw std::runtime_error("Nonfinite FP8 reference");
    unsigned mb=e5?2:3, mask=e5?31:15, e=(raw/(1u<<mb))&mask, m=raw% (1u<<mb);
    int bias=e5?15:7;
    double value=e?std::ldexp(1.0+double(m)/(1u<<mb),int(e)-bias):std::ldexp(double(m)/(1u<<mb),1-bias);
    return int64_t(std::ldexp(raw>=128?-value:value,int(q(e5))));
}
static Wide reassemble(const Digits& d){Wide v=0;for(unsigned i=0;i<5;++i)v+=Wide(d[i])*(Wide(1)<<(7*i));return v;}
static std::string str(Wide x){bool neg=x<0;UWide u=neg?UWide(0)-UWide(x):UWide(x);std::string s;do{s.push_back(char('0'+u%10));u/=10;}while(u);if(neg)s.push_back('-');std::reverse(s.begin(),s.end());return s;}
static bool decode_test(){bool ok=true;for(bool f:{false,true}){unsigned good=0,bad=0,special=0;for(unsigned i=0;i<256;++i){if(!finite(i,f)){++special;try{decode(i,f);++bad;}catch(const std::runtime_error&){}continue;}if(reassemble(decode(i,f))!=reference(i,f))++bad;else ++good;}std::printf("decode %s finite=%u special_rejected=%u mismatch=%u\n",f?"E5M2":"E4M3FN",good,special,bad);ok &= bad==0;}return ok;}
#ifndef CPU_ONLY
static void check(aclError e,const char* where){if(e!=ACL_SUCCESS)throw std::runtime_error(std::string(where)+": "+std::to_string(e));}
struct Engine{
    int dev; bool initialized=false, selected=false; aclrtStream stream=nullptr;void *a=nullptr,*b=nullptr,*c=nullptr;
    explicit Engine(int d):dev(d){}
    void start(){check(aclInit(nullptr),"aclInit");initialized=true;check(aclrtSetDevice(dev),"aclrtSetDevice");selected=true;check(aclrtCreateStream(&stream),"aclrtCreateStream");check(aclrtMalloc(&a,MAX_BLOCKS*AS,ACL_MEM_MALLOC_NORMAL_ONLY),"alloc A");check(aclrtMalloc(&b,MAX_BLOCKS*BS,ACL_MEM_MALLOC_NORMAL_ONLY),"alloc B");check(aclrtMalloc(&c,MAX_BLOCKS*CS*4,ACL_MEM_MALLOC_NORMAL_ONLY),"alloc C");}
    void run(const int8_t* x,const int8_t* y,int32_t* z,unsigned blocks){size_t ab=blocks*AS,bb=blocks*BS,cb=blocks*CS*sizeof(int32_t);check(aclrtMemcpy(a,ab,x,ab,ACL_MEMCPY_HOST_TO_DEVICE),"copy A");check(aclrtMemcpy(b,bb,y,bb,ACL_MEMCPY_HOST_TO_DEVICE),"copy B");check(aclrtMemcpy(c,cb,z,cb,ACL_MEMCPY_HOST_TO_DEVICE),"poison C");launch_int8_tiles(a,b,c,blocks,stream);check(aclrtSynchronizeStream(stream),"synchronize Cube");check(aclrtMemcpy(z,cb,c,cb,ACL_MEMCPY_DEVICE_TO_HOST),"copy C");}
    ~Engine(){if(stream)aclrtSynchronizeStream(stream);if(c)aclrtFree(c);if(b)aclrtFree(b);if(a)aclrtFree(a);if(stream)aclrtDestroyStream(stream);if(selected)aclrtResetDevice(dev);if(initialized)aclFinalize();}
};
#else
struct Engine{explicit Engine(int){}void start(){}void run(const int8_t*x,const int8_t*y,int32_t*z,unsigned blocks){for(unsigned b=0;b<blocks;++b)for(unsigned i=0;i<TM;++i)for(unsigned j=0;j<TN;++j){int32_t s=0;for(unsigned k=0;k<TK;++k)s+=int32_t(x[b*AS+i*TK+k])*y[b*BS+k*TN+j];z[b*CS+i*TN+j]=s;}}};
#endif
struct Job{unsigned pa,pb,offset;};
static uint8_t maxcode(bool f){return f?0x7b:0x7e;}
static bool test(Engine& engine,bool fa,bool fb,unsigned K,unsigned blocks,unsigned pattern,std::mt19937& rng){
    const unsigned da=planes(fa),db=planes(fb);
    std::vector<uint8_t> a(TM*K),b(K*TN);
    auto random=[&](bool f){uint8_t x;do{x=uint8_t(rng()&255);}while(!finite(x,f));return x;};
    for(unsigned i=0;i<TM;++i)for(unsigned k=0;k<K;++k){uint8_t x=random(fa);if(pattern==1)x=maxcode(fa);if(pattern==2)x=(k%3==0?maxcode(fa):k%3==1?1:uint8_t(maxcode(fa)|128));if(pattern==3)x=uint8_t(k%2?0x80:0);a[i*K+k]=x;}
    for(unsigned k=0;k<K;++k)for(unsigned j=0;j<TN;++j){uint8_t x=random(fb);if(pattern==1)x=maxcode(fb);if(pattern==2)x=(k%3==1?1:maxcode(fb));if(pattern==3)x=maxcode(fb);b[k*TN+j]=x;}
    std::vector<Digits> ad(a.size()),bd(b.size());std::vector<int64_t> ar(a.size()),br(b.size());
    for(size_t i=0;i<a.size();++i){ad[i]=decode(a[i],fa);ar[i]=reference(a[i],fa);}
    for(size_t i=0;i<b.size();++i){bd[i]=decode(b[i],fb);br[i]=reference(b[i],fb);}
    std::vector<Job> jobs;for(unsigned off=0;off<K;off+=TK)for(unsigned i=0;i<da;++i)for(unsigned j=0;j<db;++j)jobs.push_back({i,j,off});
    std::vector<Wide> out(CS,0);std::vector<int8_t> xa(blocks*AS),xb(blocks*BS);std::vector<int32_t> z(blocks*CS);
    size_t partialBad=0,finalBad=0,partialCount=0;double deviceSeconds=0;
    auto begin=std::chrono::steady_clock::now();
    for(size_t pos=0;pos<jobs.size();pos+=blocks){unsigned nb=unsigned(std::min<size_t>(blocks,jobs.size()-pos));std::fill(xa.begin(),xa.end(),0);std::fill(xb.begin(),xb.end(),0);std::fill(z.begin(),z.end(),0x5a5a5a5a);
        for(unsigned t=0;t<nb;++t){auto job=jobs[pos+t];for(unsigned k=0;k<TK&&job.offset+k<K;++k){for(unsigned i=0;i<TM;++i)xa[t*AS+i*TK+k]=ad[i*K+job.offset+k][job.pa];for(unsigned j=0;j<TN;++j)xb[t*BS+k*TN+j]=bd[(job.offset+k)*TN+j][job.pb];}}
        auto s=std::chrono::steady_clock::now();engine.run(xa.data(),xb.data(),z.data(),nb);deviceSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count();
        for(unsigned t=0;t<nb;++t){auto job=jobs[pos+t];Wide weight=Wide(1)<<(7*(job.pa+job.pb));for(unsigned i=0;i<TM;++i)for(unsigned j=0;j<TN;++j){int32_t expected=0;for(unsigned k=0;k<TK;++k)expected+=int32_t(xa[t*AS+i*TK+k])*xb[t*BS+k*TN+j];int32_t got=z[t*CS+i*TN+j];++partialCount;if(expected!=got){if(partialBad<4)std::printf(" partial mismatch Koff=%u plane=%u,%u row=%u col=%u expected=%d got=%d\n",job.offset,job.pa,job.pb,i,j,expected,got);++partialBad;}out[i*TN+j]+=Wide(got)*weight;}}
    }
    for(unsigned i=0;i<TM;++i)for(unsigned j=0;j<TN;++j){Wide expected=0;for(unsigned k=0;k<K;++k)expected+=Wide(ar[i*K+k])*Wide(br[k*TN+j]);if(out[i*TN+j]!=expected){if(finalBad<4)std::printf(" final mismatch row=%u col=%u expected=%s got=%s\n",i,j,str(expected).c_str(),str(out[i*TN+j]).c_str());++finalBad;}}
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::printf("%s x %s K=%u pattern=%u tiles=%zu partial_mismatch=%zu/%zu final_mismatch=%zu/%u exponent=-%u transfer_launch_seconds=%.6f checked_wall_seconds=%.6f %s\n",fa?"E5M2":"E4M3FN",fb?"E5M2":"E4M3FN",K,pattern,jobs.size(),partialBad,partialCount,finalBad,CS,q(fa)+q(fb),deviceSeconds,wall,partialBad||finalBad?"FAIL":"PASS");
    std::printf("  C[0,0] exact = %s * 2^(-%u)\n",str(out[0]).c_str(),q(fa)+q(fb));std::fflush(stdout);
    return partialBad==0&&finalBad==0;
}
int main(int argc,char**argv){try{
    int device=argc>1?std::stoi(argv[1]):0,K=argc>2?std::stoi(argv[2]):256,rounds=argc>3?std::stoi(argv[3]):1,blocks=argc>4?std::stoi(argv[4]):20;
    if(device<0||K<1||K>8192||rounds<1||blocks<1||blocks>int(MAX_BLOCKS))throw std::runtime_error("usage: demo [device>=0] [K 1..8192] [rounds>=1] [blocks 1..20]");
#ifdef CPU_ONLY
    std::puts("CPU-only emulation of partial INT8 tiles; NOT NPU execution");
#else
    std::printf("NPU device %d; max %d Cube blocks per launch; CPU decode/reconstruction\n",device,blocks);
#endif
    bool ok=decode_test();Engine e(device);e.start();std::mt19937 rng(910);
    for(int r=0;r<rounds;++r)for(bool fa:{false,true})for(bool fb:{false,true})for(unsigned p=0;p<4;++p)ok &= test(e,fa,fb,K,blocks,p,rng);
    std::printf("FINAL: %s\n",ok?"PASS (all checked integer results exact)":"FAIL");return ok?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}}
