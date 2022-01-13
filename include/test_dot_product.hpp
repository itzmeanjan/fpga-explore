#pragma once
#include "utils.hpp"

#if defined(dot_product_method_0)
#include "dot_product/method_0.hpp"
#pragma message("Compiling dot product method_0 kernel")
#elif defined(dot_product_method_1)
#include "dot_product/method_1.hpp"
#pragma message("Compiling dot product method_1 kernel")
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

#if defined(dot_product_method_0) || defined(dot_product_method_1)

sycl::cl_ulong
#if defined(dot_product_method_0)
method_0
#elif defined(dot_product_method_1)
method_1
#endif
  (sycl::queue& q, size_t in_len, size_t wg_size)
{
  sycl::uint* in_a_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint) * in_len, q));
  sycl::uint* in_a_d = static_cast<sycl::uint*>(
    sycl::malloc_device(sizeof(sycl::uint) * in_len, q));
  sycl::uint* in_b_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint) * in_len, q));
  sycl::uint* in_b_d = static_cast<sycl::uint*>(
    sycl::malloc_device(sizeof(sycl::uint) * in_len, q));
  sycl::uint* out_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint), q));
  sycl::uint* out_d =
    static_cast<sycl::uint*>(sycl::malloc_device(sizeof(sycl::uint), q));

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
#endif
    (q,
     in_a_d,
     in_len,
     in_b_d,
     in_len,
     out_d,
     wg_size,
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

  sycl::free(in_a_h, q);
  sycl::free(in_a_d, q);
  sycl::free(in_b_h, q);
  sycl::free(in_b_d, q);
  sycl::free(out_h, q);
  sycl::free(out_d, q);

  return ts;
}

#endif

}
