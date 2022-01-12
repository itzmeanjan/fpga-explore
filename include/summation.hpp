#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
sycl::event
method_0(sycl::queue& q,
         const sycl::uint* in,
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
    h.parallel_for<class kernelSummationMethod0>(
      sycl::nd_range<1>{ sycl::range<1>{ in_len }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        // which input element to (atomically)  add
        const size_t idx = it.get_global_linear_id();

        sycl::ext::oneapi::atomic_ref<
          sycl::uint,
          sycl::ext::oneapi::memory_order::relaxed,
          sycl::ext::oneapi::memory_scope::device,
          sycl::access::address_space::ext_intel_global_device_space>
          out_ref{ *out };
        out_ref.fetch_add(*(in + idx));
        // atomically update global sum holder memory location with
        // relaxed memory ordering
      });
  });
}

sycl::event
method_1(sycl::queue& q,
         const sycl::uint* in,
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

    h.parallel_for<class kernelSummationMethod1>(
      sycl::nd_range<1>{ sycl::range<1>{ in_len }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
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
        scratch_ref.fetch_add(*(in + glb_idx));

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
            out_ref{ *out };
          out_ref.fetch_add(scratch[0]);
        }
      });
  });
}

sycl::event
method_2(sycl::queue& q,
         const sycl::uint* in,
         size_t in_len,
         sycl::uint* const out,
         size_t out_len,
         size_t wg_size,
         std::vector<sycl::event> evts)
{
  assert(wg_size <= in_len);
  // so that all work-groups are of same size
  assert(in_len % wg_size == 0);
  // must supply atleast 2 input elements to sum together
  assert(in_len >= 2);
  // power of 2 test i.e. input length must be power of 2
  assert((in_len & (in_len - 1)) == 0);

  const size_t total_work_items = in_len >> 1;
  const size_t rev_wg_size =
    wg_size <= total_work_items ? wg_size : total_work_items;

  // each work-group local summation is stored in output allocation
  assert(out_len == in_len / rev_wg_size);

  return q.submit([&](sycl::handler& h) {
    sycl::accessor<sycl::uint,
                   1,
                   sycl::access_mode::read_write,
                   sycl::target::local>
      scratch{ sycl::range<1>{ rev_wg_size }, h };

    h.depends_on(evts);
    h.parallel_for<class kernelSummationMethod2>(
      sycl::nd_range<1>{ sycl::range<1>{ total_work_items },
                         sycl::range<1>{ rev_wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t glb_idx = it.get_global_linear_id();
        const size_t loc_idx = it.get_local_linear_id();
        const size_t wg_size = it.get_group_range(0);
        const size_t wg_idx = it.get_group().get_linear_id();

        sycl::group<1> grp = it.get_group();
        const size_t in_offset = glb_idx << 1;

        scratch[loc_idx] = *(in + in_offset + 0) + *(in + in_offset + 1);

        sycl::group_barrier(grp);

        size_t active_work_items = wg_size >> 1;
        while (active_work_items > 0) {
          if (active_work_items > loc_idx) {
            scratch[loc_idx] =
              scratch[loc_idx] + scratch[loc_idx + active_work_items];
          }

          active_work_items >>= 1;
          sycl::group_barrier(grp);
        }

        if (loc_idx == 0) {
          *(out + wg_idx) = scratch[0];
        }
      });
  });
}
}
