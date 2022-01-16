#pragma once
#include "utils.hpp"

#if defined(mat_mul_method_0)
#include "mat_mul/method_0.hpp"
#pragma message("Compiling matrix multiplication method_0 kernel")
#else
#pragma message(                                                               \
  "Specify which kernel(s) to compile, when invoking `make` utility")
#endif

namespace test_mat_mul {

void
sequential_mat_mul(const sycl::uint* in_a,
                   size_t rows_a,
                   size_t cols_a,
                   const sycl::uint* in_b,
                   size_t rows_b,
                   size_t cols_b,
                   sycl::uint* const out)
{
  assert(cols_a == rows_b);

  for (size_t i = 0; i < rows_a; i++) {
    for (size_t j = 0; j < cols_b; j++) {
      sycl::uint tmp = 0;
      for (size_t k = 0; k < cols_a; k++) {
        tmp += *(in_a + i * cols_a + k) * *(in_b + k * cols_b + j);
      }
      *(out + i * cols_b + j) = tmp;
    }
  }
}

#if defined(mat_mul_method_0)

sycl::cl_ulong

#if defined(mat_mul_method_0)
method_0
#endif

  (sycl::queue& q,
   size_t rows_a,
   size_t cols_a,
   size_t rows_b,
   size_t cols_b,
   size_t wg_size_x,
   size_t wg_size_y)
{
  size_t in_a_size = sizeof(sycl::uint) * rows_a * cols_a;
  size_t in_b_size = sizeof(sycl::uint) * rows_b * cols_b;
  size_t out_size = sizeof(sycl::uint) * rows_a * cols_b;

  sycl::uint* in_a_h = static_cast<sycl::uint*>(std::malloc(in_a_size));
  sycl::uint* in_b_h = static_cast<sycl::uint*>(std::malloc(in_b_size));
  sycl::uint* out_h = static_cast<sycl::uint*>(std::malloc(out_size));
  sycl::uint* out_h_chk = static_cast<sycl::uint*>(std::malloc(out_size));

  sycl::uint* in_a_d = sycl::malloc_device<sycl::uint>(rows_a * cols_a, q);
  sycl::uint* in_b_d = sycl::malloc_device<sycl::uint>(rows_b * cols_b, q);
  sycl::uint* out_d = sycl::malloc_device<sycl::uint>(rows_a * cols_b, q);

  random_fill(in_a_h, rows_a * cols_a);
  random_fill(in_b_h, rows_b * cols_b);

  sycl::event evt_0 = q.memcpy(in_a_d, in_a_h, in_a_size);
  sycl::event evt_1 = q.memcpy(in_b_d, in_b_h, in_b_size);
  sycl::event evt_2 = mat_mul::

#if defined(mat_mul_method_0)
    method_0
#endif

    (q,
     in_a_d,
     rows_a,
     cols_a,
     in_b_d,
     rows_b,
     cols_b,
     out_d,
     wg_size_x,
     wg_size_y,
     { evt_0, evt_1 });
  sycl::event evt_3 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt_2);
    h.memcpy(out_h, out_d, out_size);
  });

  evt_3.wait();

  // compute matrix multiplication on host ( O(n2) implementation ! )
  sequential_mat_mul(in_a_h, rows_a, cols_a, in_b_h, rows_b, cols_b, out_h_chk);

  // check for correctness of result computed on fpga !
  for (size_t i = 0; i < rows_a; i++) {
    for (size_t j = 0; j < cols_b; j++) {
      assert(*(out_h + i + j) == *(out_h_chk + i + j));
    }
  }

  sycl::cl_ulong ts = time_event(evt_2);

  // deallocate host allocated memory
  std::free(in_a_h);
  std::free(in_b_h);
  std::free(out_h);
  std::free(out_h_chk);

  // deallocate sycl runtime managed memory
  sycl::free(in_a_d, q);
  sycl::free(in_b_d, q);
  sycl::free(out_d, q);

  return ts;
}

#endif

}
