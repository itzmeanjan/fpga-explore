#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
class kernelSummationMethod4;

sycl::event
method_4(sycl::queue& q,
         sycl::uint* const in,
         size_t in_len,
         sycl::uint* const out,
         std::vector<sycl::event> evts)
{
  // must supply atleast 2 input elements to sum together
  assert(in_len >= 2);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    // single work-item kernel
    h.single_task<kernelSummationMethod4>([=
    ]() [[intel::kernel_args_restrict]] {
      sycl::device_ptr<sycl::uint> in_ptr{ in };
      sycl::device_ptr<sycl::uint> out_ptr{ out };

      [[intel::fpga_register]] sycl::uint tmp[8];

#pragma unroll 8 // fully unrolled; parallelized
      for (size_t i = 0; i < 8; i++) {
        tmp[i] = 0;
      }

#pragma unroll 8 // partially unrolled
      for (size_t i = 0; i < in_len; i++) {
        tmp[i % 8] += in_ptr[i];
      }

      out_ptr[0] =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    });
  });
}
}
