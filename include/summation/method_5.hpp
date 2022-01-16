#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
class kernelSummationMethod5;

sycl::event
method_5(sycl::queue& q,
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
    h.single_task<kernelSummationMethod5>([=
    ]() [[intel::kernel_args_restrict]] {
      sycl::device_ptr<sycl::uint> in_ptr{ in };
      sycl::device_ptr<sycl::uint> out_ptr{ out };

      [[intel::fpga_register]] sycl::uint tmp = 0;

#pragma unroll // partially unrolled
      for (size_t i = 0; i < in_len; i++) {
        tmp += in_ptr[i];
      }

      out_ptr[0] = tmp;
    });
  });
}
}
