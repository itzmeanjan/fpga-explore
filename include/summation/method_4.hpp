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

      [[intel::fpga_register]] sycl::uint tmp_0[8];
      [[intel::fpga_register]] sycl::uint tmp_1[8];

#pragma unroll 8 // fully unrolled; parallelized
      for (size_t i = 0; i < 8; i++) {
        tmp_0[i] = 0;
        tmp_1[i] = 0;
      }

      const size_t upto = in_len >> 1;
#pragma unroll 8 // partially unrolled
      for (size_t i = 0; i < upto; i++) {
        tmp_0[i % 8] += in_ptr[i];
        tmp_1[i % 8] += in_ptr[upto + i];
      }

      out_ptr[0] = tmp_0[0] + tmp_0[1] + tmp_0[2] + tmp_0[3] + tmp_0[4] +
                   tmp_0[5] + tmp_0[6] + tmp_0[7] + tmp_1[0] + tmp_1[1] +
                   tmp_1[2] + tmp_1[3] + tmp_1[4] + tmp_1[5] + tmp_1[6] +
                   tmp_1[7];
    });
  });
}
}
