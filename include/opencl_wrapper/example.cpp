/*-----example.cpp demonstrates the upgraded opencl wrapper
/compile: g++ -std=c++17 example.cpp -lOpenCL -o example
 *
 */
#include<pipeline.hpp>
#include<cmath>
#include<iostream>
#include<numeric>
//--------OpenCl kernel source-----------
//---------------------------------------------------------------------------------------------------------------
static const char *SRC=R"cl(
__kernel void vec_add(__global const float* a,
__global const float* b,
_global const float* c,
int n){
int i= get_global_id(0);
if(i<n) c[i]=a[i]+b[i];
}
)cl";
int main(){
	const int N = 1024;
	//---------------Build pipeline from inline source-----------------------------------
	auto p=Pipeline::from_source(SRC,{"vec_add"}, "-cl-std=CL3.0",
			CL_QUEUE_PROFILING_ENABLE);
	std::cout <<"Device: "<<Platform ::get_device_name(p.device)<<"\n";
	platform::print_device_info(p.device);
	//Prepare host data----------------------------------------------------------------------------
	std::vector<float>ha(N),hb(N),hc(N, 0.f);
	std::iota(ha.begin(), ha.end(),0.f); //ha = [0,1,2,...,1023]
	std::iota(hb.begin(),hb.end(),100.f);//hb=[100,101,...1123]
	//Allocate device buffers----------------------------------------------------------------------
	auto da=p.make_buffer(ha, CL_MEM_READ_ONLY);
	auto db=p.make_buffer(hb, CL_MEM_READ_ONLY);
	auto dc=p.make_buffer(N* sizeof(float),CL_MEM_WRITE_ONLY);
	//----------DIspatch-------------------------------------------------------------------------------------
	dispatch::NDRange range{dispatch::round_up (N, 64), 64};
	dispatch::Event ev{dispatch::run (p.queue, p.kernel("vec_add"), range, da.handle, db.handle,dc.handle,N)};
	ev.wait();
	std::cout <<"Elapsed: "<< ev.elapsed_ms()<<"ms\n";
	// -----------Read back-----------------------------------------------------------------------------------
	dc.read(p.queue, hc);
	//-----------------------Verify--------------------------------------------------------------------------------
	bool ok=true;
	for(int i=0;i<N;++i){
		if(std::abs(hc[i]-(ha[i]+hb[i]))>1e-5f){
			ok=false;
			break;
		}
	}
	std::cout <<(ok? "PASS\n":"FAIL\n");
	return 0;
}
