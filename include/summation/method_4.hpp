#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
class kernelSummationMethod4;

sycl::event
method_4(sycl::queue& q,
         const sycl::uint* in,
         size_t in_len,
         sycl::uint* const out,
         std::vector<sycl::event> evts)
{
  // must supply atleast 2 input elements to sum together
  assert(in_len >= 2);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    // single work-item kernel
    h.single_task<kernelSummationMethod4>([=]() {
      sycl::uint tmp = 0;

      for (size_t i = 0; i < in_len; i++) {
        tmp += *(in + i);
      }

      *out = tmp;
    });
  });
}
}
