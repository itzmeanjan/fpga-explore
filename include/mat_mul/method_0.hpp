#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace matmul {
class kernelMatMulMethod0;

sycl::event
method_0(sycl::queue& q,
         sycl::uint* const in_a,
         size_t rows_a,
         size_t cols_a,
         sycl::uint* const in_b,
         size_t rows_b,
         size_t cols_b,
         sycl::uint* const out,
         size_t wg_size_x,
         size_t wg_size_y,
         std::vector<sycl::event> evts)
{
  assert(cols_a == rows_b);
  assert(rows_a % wg_size_x == 0);
  assert(cols_b % wg_size_y == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.parallel_for<kernelMatMulMethod0>(
      sycl::nd_range<2>{ sycl::range<2>{ rows_a, cols_b },
                         sycl::range<2>{ wg_size_x, wg_size_y } },
      [=](sycl::nd_item<2> it) [[intel::kernel_args_restrict,
                                 intel::num_simd_work_items(8),
                                 sycl::reqd_work_group_size(1, 1, 32)]] {
        sycl::device_ptr<sycl::uint> in_a_ptr{ in_a };
        sycl::device_ptr<sycl::uint> in_b_ptr{ in_b };
        sycl::device_ptr<sycl::uint> out_ptr{ out };

        const size_t r = it.get_global_id(0);
        const size_t c = it.get_global_id(1);

        sycl::uint tmp = 0;
        for (size_t i = 0; i < cols_a; i++) {
          tmp += in_a_ptr[r * cols_a + i] * in_b_ptr[i * cols_b + c];
        }

        out[r * cols_b + c] = tmp;
      });
  });
}
}
