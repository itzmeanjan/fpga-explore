#pragma once
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

void
make_queue(void** wq)
{

#if defined(TARGET_CPU)
  sycl::cpu_selector d_sel{};
#elif defined(TARGET_GPU)
  sycl::gpu_selector d_sel{};
#elif defined(TARGET_FPGA_EMU)
  sycl::ext::intel::fpga_emulator_selector d_sel{};
#elif defined(TARGET_FPGA)
  sycl::ext::intel::fpga_selector d_sel{};
#else
  sycl::default_selector d_sel{};
#endif

  sycl::device d{ d_sel };
  sycl::context c{ d };
  sycl::queue* q = new sycl::queue{ c, d };

  *wq = q;
}
