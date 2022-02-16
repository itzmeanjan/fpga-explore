#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace dot_product {
class kernelDotProductMethod4;

sycl::event
method_4(sycl::queue& q,
         sycl::uint* in_a,
         size_t in_a_len,
         sycl::uint* in_b,
         size_t in_b_len,
         sycl::uint* const out,
         std::vector<sycl::event> evts)
{
  // both input vectors should have same number of elements
  assert(in_a_len == in_b_len);
  assert(in_a_len >= 16);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.single_task<kernelDotProductMethod4>([=
    ]() [[intel::kernel_args_restrict]] {
      // inform compiler that these pointers point
      // to device memory allocation ( USM based )
      sycl::device_ptr<sycl::uint> in_a_ptr{ in_a };
      sycl::device_ptr<sycl::uint> in_b_ptr{ in_b };
      sycl::device_ptr<sycl::uint> out_ptr{ out };

      // so that each 32 -bit unsigned integer element can be accessed in
      // parallel, in stall free manner
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(4),
        intel::numbanks(16)]] sycl::uint loader_a[16];
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(4),
        intel::numbanks(16)]] sycl::uint loader_b[16];
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(4),
        intel::numbanks(16)]] sycl::uint tmp_sum[16];

#pragma unroll 16
      for (size_t i = 0; i < 16; i++) {
        tmp_sum[i] = 0;
      }

      [[intel::ivdep]] for (size_t i = 0; i < in_a_len; i += 16)
      {
#pragma unroll 16 // 512 -bit burst coalesced loading from global memory
        for (size_t j = 0; j < 16; j++) {
          loader_a[j] = in_a_ptr[i + j];
        }
#pragma unroll 16 // 512 -bit burst coalesced loading from global memory
        for (size_t j = 0; j < 16; j++) {
          loader_b[j] = in_b_ptr[i + j];
        }

#pragma unroll 16
        for (size_t j = 0; j < 16; j++) {
          tmp_sum[j] += loader_a[j] * loader_b[j];
        }
      }

      // finally write dot product back to global memory
      out_ptr[0] = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3] +
                   tmp_sum[4] + tmp_sum[5] + tmp_sum[6] + tmp_sum[7] +
                   tmp_sum[8] + tmp_sum[9] + tmp_sum[10] + tmp_sum[11] +
                   tmp_sum[12] + tmp_sum[13] + tmp_sum[14] + tmp_sum[15];
    });
  });
}
}
