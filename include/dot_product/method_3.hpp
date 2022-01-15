#pragma once
#include <CL/sycl.hpp>
#include <cassert>

// VECTOR_WIDTH can take value from set {2, 4, 8, 16}
// while VECTOR_WIDTH_LOG2 will be log2(VECTOR_WIDTH)
#define VECTOR_WIDTH 4
#define VECTOR_WIDTH_LOG2 2

#if !((VECTOR_WIDTH == 2 && VECTOR_WIDTH_LOG2 == 1) ||                         \
      (VECTOR_WIDTH == 4 && VECTOR_WIDTH_LOG2 == 2) ||                         \
      (VECTOR_WIDTH == 8 && VECTOR_WIDTH_LOG2 == 3) ||                         \
      (VECTOR_WIDTH == 16 && VECTOR_WIDTH_LOG2 == 4))
#error                                                                         \
  "VECTOR_WIDTH can only be from {2, 4, 8, 16}, while VECTOR_WIDTH_LOG2 must be logarithm base 2 of VECTOR_WIDTH"
#endif

namespace dot_product {
class kernelDotProductMethod3;

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
  // in each iteration 2/ 4/ 8/ 16 consequtive input elements
  // can be loaded from global memory
  assert(in_a_len % VECTOR_WIDTH == 0);
  // input vector is of length power of 2
  assert((in_a_len & (in_a_len - 1)) == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.single_task<kernelDotProductMethod3>([=
    ]() [[intel::kernel_args_restrict]] {
      const size_t upto = in_a_len >> VECTOR_WIDTH_LOG2;

      sycl::device_ptr<sycl::uint> in_a_ptr{ in_a };
      sycl::device_ptr<sycl::uint> in_b_ptr{ in_b };
      sycl::device_ptr<sycl::uint> out_ptr{ out };

      sycl::uint tmp = 0;

      for (size_t i = 0; i < upto; i++) {
#if VECTOR_WIDTH == 2
#pragma message("Using vector width 2 for dot_product::method_3 kernel")

        sycl::uint2 a;
        sycl::uint2 b;
        sycl::uint2 c;
#elif VECTOR_WIDTH == 4
#pragma message("Using vector width 4 for dot_product::method_3 kernel")

        sycl::uint4 a;
        sycl::uint4 b;
        sycl::uint4 c;
#elif VECTOR_WIDTH == 8
#pragma message("Using vector width 8 for dot_product::method_3 kernel")

        sycl::uint8 a;
        sycl::uint8 b;
        sycl::uint8 c;
#elif VECTOR_WIDTH == 16
#pragma message("Using vector width 16 for dot_product::method_3 kernel")

        sycl::uint16 a;
        sycl::uint16 b;
        sycl::uint16 c;
#endif

        a.load(i, in_a_ptr);
        b.load(i, in_b_ptr);

        c = a * b;

#if VECTOR_WIDTH == 2
        tmp += (c.x() + c.y());
#elif VECTOR_WIDTH == 4
        tmp += (c.x() + c.y() + c.z() + c.w());
#elif VECTOR_WIDTH == 8
        tmp += (c.s0() + c.s1() + c.s2() + c.s3() + c.s4() + c.s5() + c.s6() +
                c.s7());
#elif VECTOR_WIDTH == 16
        tmp += (c.s0() + c.s1() + c.s2() + c.s3() + c.s4() + c.s5() + c.s6() +
                c.s7() + c.s8() + c.s9() + c.sA() + c.sB() + c.sC() + c.sD() +
                c.sE() + c.sF());
#endif
      }

      out_ptr[0] = tmp;
    });
  });
}
}
