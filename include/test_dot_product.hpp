#pragma once
#include "utils.hpp"

#if defined(dot_product_method_0)
#include "dot_product/method_0.hpp"
#pragma message("Compiling dot product method_0 kernel")
#elif defined(dot_product_method_1)
#include "dot_product/method_1.hpp"
#pragma message("Compiling dot product method_1 kernel")
#elif defined(dot_product_method_2)
#include "dot_product/method_2.hpp"
#pragma message("Compiling dot product method_2 kernel")
#elif defined(dot_product_method_3)
#include "dot_product/method_3.hpp"
#pragma message("Compiling dot product method_3 kernel")
#elif defined(dot_product_method_4)
#include "dot_product/method_4.hpp"
#pragma message("Compiling dot product method_4 kernel")
#else
#pragma message(                                                               \
  "Specify which kernel(s) to compile, when invoking `make` utility")
#endif

namespace test_dot_product {

void
seq_dot_product(const sycl::uint* in_a,
                size_t in_a_len,
                const sycl::uint* in_b,
                size_t in_b_len,
                sycl::uint* const out)
{
  assert(in_a_len == in_b_len);

  sycl::uint tmp = 0;
  for (size_t i = 0; i < in_a_len; i++) {
    tmp += *(in_a + i) * *(in_b + i);
  }
  *out = tmp;
}

#if defined(dot_product_method_0) || defined(dot_product_method_1) ||          \
  defined(dot_product_method_2) || defined(dot_product_method_3) ||            \
  defined(dot_product_method_4)

sycl::cl_ulong
#if defined(dot_product_method_0)
method_0
#elif defined(dot_product_method_1)
method_1
#elif defined(dot_product_method_2)
method_2
#elif defined(dot_product_method_3)
method_3
#elif defined(dot_product_method_4)
method_4
#endif
  (sycl::queue& q, size_t in_len, size_t wg_size)
{
  sycl::uint* in_a_h =
    static_cast<sycl::uint*>(std::malloc(sizeof(sycl::uint) * in_len));
  mem_alloc_check<sycl::uint>(in_a_h, in_len, mem_alloc::HOST);

  sycl::uint* in_a_d = sycl::malloc_device<sycl::uint>(in_len, q);
  mem_alloc_check<sycl::uint>(in_a_d, in_len, mem_alloc::DEVICE);

  sycl::uint* in_b_h =
    static_cast<sycl::uint*>(std::malloc(sizeof(sycl::uint) * in_len));
  mem_alloc_check<sycl::uint>(in_b_h, in_len, mem_alloc::HOST);

  sycl::uint* in_b_d = sycl::malloc_device<sycl::uint>(in_len, q);
  mem_alloc_check<sycl::uint>(in_b_d, in_len, mem_alloc::DEVICE);

  sycl::uint* out_h = static_cast<sycl::uint*>(std::malloc(sizeof(sycl::uint)));
  mem_alloc_check<sycl::uint>(out_h, 1, mem_alloc::HOST);

  sycl::uint* out_d = sycl::malloc_device<sycl::uint>(1, q);
  mem_alloc_check<sycl::uint>(out_d, 1, mem_alloc::DEVICE);

  random_fill(in_a_h, in_len);
  random_fill(in_b_h, in_len);

  sycl::event evt_0 = q.memcpy(in_a_d, in_a_h, sizeof(sycl::uint) * in_len);
  sycl::event evt_1 = q.memcpy(in_b_d, in_b_h, sizeof(sycl::uint) * in_len);
  sycl::event evt_2 = q.memset(out_d, 0, sizeof(sycl::uint));

  sycl::event evt_3 = dot_product::
#if defined(dot_product_method_0)
    method_0
#elif defined(dot_product_method_1)
    method_1
#elif defined(dot_product_method_2)
    method_2
#elif defined(dot_product_method_3)
    method_3
#elif defined(dot_product_method_4)
    method_4
#endif
    (q,
     in_a_d,
     in_len,
     in_b_d,
     in_len,
     out_d,
#if defined(dot_product_method_0) || defined(dot_product_method_1)
     wg_size,
#endif
     { evt_0, evt_1, evt_2 });
  sycl::event evt_4 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt_3);
    h.memcpy(out_h, out_d, sizeof(sycl::uint));
  });

  evt_4.wait();

  // to be asserted against it
  sycl::uint out_cmp = 0;
  seq_dot_product(in_a_h, in_len, in_b_h, in_len, &out_cmp);

  assert(*out_h == out_cmp);

  sycl::cl_ulong ts = time_event(evt_3);

  std::free(in_a_h);
  std::free(in_b_h);
  std::free(out_h);

  sycl::free(in_a_d, q);
  sycl::free(in_b_d, q);
  sycl::free(out_d, q);

  return ts;
}

#endif

}
