#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
class kernelSummationMethod4;

sycl::event
method_4(sycl::queue& q,
         const sycl::uint* in,
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
      [[intel::fpga_register]] sycl::uint tmp[4] = { 0, 0, 0, 0 };

#pragma unroll 4
      for (size_t i = 0; i < in_len; i++) {
        tmp[i % 4] += *(in + i);
      }

      *out = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    });
  });
}
}
