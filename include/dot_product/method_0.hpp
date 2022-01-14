#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace dot_product {
class kernelDotProductMethod0;

sycl::event
method_0(sycl::queue& q,
         sycl::uint* in_a,
         size_t in_a_len,
         sycl::uint* in_b,
         size_t in_b_len,
         sycl::uint* const out,
         size_t wg_size,
         std::vector<sycl::event> evts)
{
  // both input vectors should have same number of elements
  assert(in_a_len == in_b_len);
  assert(wg_size <= in_a_len);
  // ensure that all dispatched work-groups are of same size
  assert(in_a_len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.parallel_for<kernelDotProductMethod0>(
      sycl::nd_range<1>{ sycl::range<1>{ in_a_len },
                         sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        sycl::device_ptr<sycl::uint> in_a_ptr{ in_a };
        sycl::device_ptr<sycl::uint> in_b_ptr{ in_b };
        sycl::device_ptr<sycl::uint> out_ptr{ out };

        const size_t idx = it.get_global_linear_id();
        sycl::uint tmp = in_a_ptr[idx] * in_b_ptr[idx];

        // atomically update dot product stored in global memory
        sycl::ext::oneapi::atomic_ref<
          sycl::uint,
          sycl::ext::oneapi::memory_order::relaxed,
          sycl::ext::oneapi::memory_scope::device,
          sycl::access::address_space::ext_intel_global_device_space>
          out_ref{ out_ptr[0] };
        out_ref.fetch_add(tmp);
      });
  });
}
}
