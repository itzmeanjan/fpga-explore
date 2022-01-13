#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace dot_product {
class kernelDotProductMethod2;

sycl::event
method_2(sycl::queue& q,
         const sycl::uint* in_a,
         size_t in_a_len,
         const sycl::uint* in_b,
         size_t in_b_len,
         sycl::uint* const out,
         std::vector<sycl::event> evts)
{
  // both input vectors should have same number of elements
  assert(in_a_len == in_b_len);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.single_task<kernelDotProductMethod2>([=]() {
      sycl::uint tmp = 0;
      for (size_t i = 0; i < in_a_len; i++) {
        tmp += *(in_a + i) * *(in_b + i);
      }

      *out = tmp;
    });
  });
}
}
