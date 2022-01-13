#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace dot_product {
sycl::event
method_3(sycl::queue& q,
         sycl::uint* in_a,
         size_t in_a_len,
         sycl::uint* in_b,
         size_t in_b_len,
         sycl::uint* const out,
         std::vector<sycl::event> evts)
{
  // both input vectors should have same number of elements
  assert(in_a_len == in_b_len);
  // just to be sure that N -many iterations can be run, where
  // in each iteration 4 consequtive input elements
  // can be loaded from global memory
  assert(in_a_len % 4 == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.single_task<class kernelDotProductMethod3>([=]() {
      const size_t upto = in_a_len >> 2;

      sycl::global_ptr<sycl::uint> in_a_ptr{ in_a };
      sycl::global_ptr<sycl::uint> in_b_ptr{ in_b };

      sycl::uint tmp = 0;

      sycl::uint4 a;
      sycl::uint4 b;
      sycl::uint4 c;

      for (size_t i = 0; i < upto; i++) {
        a.load(i << 2, in_a_ptr);
        b.load(i << 2, in_b_ptr);

        c = a * b;

        tmp += c.x() + c.y() + c.z() + c.w();
      }

      *out = tmp;
    });
  });
}
}
