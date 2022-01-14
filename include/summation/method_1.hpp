#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
class kernelSummationMethod1;

sycl::event
method_1(sycl::queue& q,
         sycl::uint* in,
         size_t in_len,
         sycl::uint* const out,
         size_t wg_size,
         std::vector<sycl::event> evts)
{
  assert(wg_size <= in_len);
  // so that all work-groups are of same size
  assert(in_len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    sycl::accessor<sycl::uint,
                   1,
                   sycl::access_mode::read_write,
                   sycl::access::target::local>
      scratch{ sycl::range<1>{ 1 }, h };

    h.parallel_for<kernelSummationMethod1>(
      sycl::nd_range<1>{ sycl::range<1>{ in_len }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        sycl::device_ptr<sycl::uint> in_ptr{ in };
        sycl::device_ptr<sycl::uint> out_ptr{ out };

        const size_t glb_idx = it.get_global_linear_id();
        const size_t loc_idx = it.get_local_linear_id();
        sycl::group<1> grp = it.get_group();

        // work-group leader initialises local memory
        if (loc_idx == 0) {
          scratch[0] = 0;
        }

        // all other work-items wait for work-group
        // leader to complete initialisation
        sycl::group_barrier(grp);

        // atomically computes work-group local summation
        // using local memory
        sycl::ext::oneapi::atomic_ref<
          sycl::uint,
          sycl::ext::oneapi::memory_order::relaxed,
          sycl::ext::oneapi::memory_scope::work_group,
          sycl::access::address_space::local_space>
          scratch_ref{ scratch[0] };
        scratch_ref.fetch_add(in_ptr[glb_idx]);

        // all work-items wait until every other work-item
        // in work-group completes work-group local summation
        sycl::group_barrier(grp);

        // only work-group leader writes work-group local
        // summation back to global memory
        if (loc_idx == 0) {
          sycl::ext::oneapi::atomic_ref<
            sycl::uint,
            sycl::ext::oneapi::memory_order::relaxed,
            sycl::ext::oneapi::memory_scope::device,
            sycl::access::address_space::ext_intel_global_device_space>
            out_ref{ out_ptr[0] };
          out_ref.fetch_add(scratch[0]);
        }
      });
  });
}
}
