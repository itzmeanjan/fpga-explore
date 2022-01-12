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
  // enabling profiling is required for timing execution of job(s)
  // using SYCL events
  sycl::queue* q =
    new sycl::queue{ c, d, sycl::property::queue::enable_profiling{} };

  *wq = q;
}

// Execution time of submitted job with nanosecond
// level of granularity
//
// Ensure that queue has profiling enabled !
sycl::cl_ulong
time_event(sycl::event evt)
{
  sycl::cl_ulong start =
    evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  sycl::cl_ulong end =
    evt.get_profiling_info<sycl::info::event_profiling::command_end>();

  return (end - start);
}
