#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
void
method_0(sycl::queue& q,
         const sycl::uint* in,
         size_t in_len,
         sycl::uint* const out,
         size_t wg_size)
{
  assert(wg_size <= in_len);
  // so that all work-groups are of same size
  assert(in_len % wg_size == 0);

  q.parallel_for<class kernelSummationMethod0>(
    sycl::nd_range<1>{ sycl::range<1>{ in_len }, sycl::range<1>{ wg_size } },
    [=](sycl::nd_item<1> it) {
      // which input element to (atomically)  add
      const size_t idx = it.get_global_linear_id();

      sycl::ext::oneapi::atomic_ref<
        sycl::uint,
        sycl::ext::oneapi::memory_order::relaxed,
        sycl::ext::oneapi::memory_scope::device,
        sycl::access::address_space::global_device_space>
        out_ref{ *out };
      out_ref.fetch_add(*(in + idx));
      // atomically update global sum holder memory location with 
      // relaxed memory ordering
    });
}
}
