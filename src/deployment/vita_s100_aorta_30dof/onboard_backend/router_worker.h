#pragma once
#include "native_router.h"
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
namespace vita_router {
class Worker {
 Router& router;
 std::mutex mutex;
 std::condition_variable ready,done;
 const float* input=nullptr;
 bool pending=false,complete=true,stop=false;
 double elapsed=0;
 std::exception_ptr failure;
 std::thread thread;
 void loop(){for(;;){const float* p;
  {std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[this]{return stop||pending;});if(stop)return;p=input;pending=false;}
  auto begin=std::chrono::steady_clock::now();std::exception_ptr error;
  try{router.run(p);}catch(...){error=std::current_exception();}
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  {std::lock_guard<std::mutex> lock(mutex);elapsed=ms;failure=error;complete=true;}done.notify_one();
 }}
public:
 explicit Worker(Router& r):router(r),thread(&Worker::loop,this){}
 ~Worker(){{std::lock_guard<std::mutex> lock(mutex);stop=true;}ready.notify_one();if(thread.joinable())thread.join();}
 Worker(const Worker&)=delete;Worker& operator=(const Worker&)=delete;
 void submit(const float* p){std::lock_guard<std::mutex> lock(mutex);if(!complete||pending||stop)throw std::runtime_error("Router worker busy");input=p;complete=false;pending=true;failure=nullptr;ready.notify_one();}
 bool wait(uint32_t timeout_ms,double& duration){std::unique_lock<std::mutex> lock(mutex);if(!done.wait_for(lock,std::chrono::milliseconds(timeout_ms),[this]{return complete;}))return false;duration=elapsed;if(failure)std::rethrow_exception(failure);return true;}
 bool idle(){std::lock_guard<std::mutex> lock(mutex);return complete;}
};
}
