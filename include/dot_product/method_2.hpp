#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace dot_product {
class kernelDotProductMethod2;

sycl::event
method_2(sycl::queue& q,
         sycl::uint* in_a,
         size_t in_a_len,
         sycl::uint* in_b,
         size_t in_b_len,
         sycl::uint* const out,
         std::vector<sycl::event> evts)
{
  // both input vectors should have same number of elements
  assert(in_a_len == in_b_len);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.single_task<kernelDotProductMethod2>([=
    ]() [[intel::kernel_args_restrict]] {
      // inform compiler that these pointers point
      // to device memory allocation ( USM based )
      sycl::device_ptr<sycl::uint> in_a_ptr{ in_a };
      sycl::device_ptr<sycl::uint> in_b_ptr{ in_b };
      sycl::device_ptr<sycl::uint> out_ptr{ out };

      [[intel::fpga_register]] sycl::uint tmp[4];

#pragma unroll 4 // fully unrolled loop
      for (size_t i = 0; i < 4; i++) {
        tmp[i] = 0;
      }

#pragma unroll 4 // partially unrolled loop
      for (size_t i = 0; i < in_a_len; i++) {
        tmp[i % 4] += in_a_ptr[i] * in_b_ptr[i];
      }

      out_ptr[0] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    });
  });
}
}
