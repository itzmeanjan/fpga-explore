#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace dot_product {
class kernelDotProductMethod1;

sycl::event
method_1(sycl::queue& q,
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
    sycl::accessor<sycl::uint,
                   1,
                   sycl::access_mode::read_write,
                   sycl::access::target::local>
      scratch{ sycl::range<1>{ 1 }, h };
    h.depends_on(evts);

    h.parallel_for<kernelDotProductMethod1>(
      sycl::nd_range<1>{ sycl::range<1>{ in_a_len },
                         sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict,
                                 intel::num_simd_work_items(8),
                                 sycl::reqd_work_group_size(1, 1, 32)]] {
        sycl::device_ptr<sycl::uint> in_a_ptr{ in_a };
        sycl::device_ptr<sycl::uint> in_b_ptr{ in_b };
        sycl::device_ptr<sycl::uint> out_ptr{ out };

        const size_t glb_idx = it.get_global_linear_id();
        const size_t loc_idx = it.get_local_linear_id();
        sycl::group<1> grp = it.get_group();

        // only work-group leader does this
        //
        // initialise local memory so that atomic addition
        // on that memory location doesn't accumulate some
        // non-zero value
        if (loc_idx == 0) {
          scratch[0] = 0;
        }

        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        sycl::uint tmp = in_a_ptr[glb_idx] * in_b_ptr[glb_idx];

        // atomically update ( work-group scope ) dot product in local memory
        sycl::ext::oneapi::atomic_ref<
          sycl::uint,
          sycl::ext::oneapi::memory_order::relaxed,
          sycl::ext::oneapi::memory_scope::work_group,
          sycl::access::address_space::local_space>
          scratch_ref{ scratch[0] };
        scratch_ref.fetch_add(tmp);

        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        // only work-group leader does this
        if (loc_idx == 0) {
          // atomically update work-group local dot-product
          // in designated location in global memory
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
