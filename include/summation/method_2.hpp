#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
class kernelSummationMethod2Phase0;
class kernelSummationMethod2Phase1;

std::vector<sycl::event>
method_2(sycl::queue& q,
         sycl::uint* const in,
         size_t in_len,
         sycl::uint* const out_itmd,
         size_t out_itmd_len,
         sycl::uint* const out_final,
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

  // each work-group local summation is stored in unique memory location
  // of output allocation
  assert(out_itmd_len == total_work_items / rev_wg_size);

  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    sycl::accessor<sycl::uint,
                   1,
                   sycl::access_mode::read_write,
                   sycl::target::local>
      scratch{ sycl::range<1>{ rev_wg_size }, h };

    h.depends_on(evts);
    h.parallel_for<kernelSummationMethod2Phase0>(
      sycl::nd_range<1>{ sycl::range<1>{ total_work_items },
                         sycl::range<1>{ rev_wg_size } },
      [=](sycl::nd_item<1> it) {
        sycl::device_ptr<sycl::uint> in_ptr{ in };
        sycl::device_ptr<sycl::uint> out_itmd_ptr{ out_itmd };

        const size_t glb_idx = it.get_global_linear_id();
        const size_t loc_idx = it.get_local_linear_id();
        const size_t wg_size = it.get_group().get_local_range(0);
        const size_t wg_idx = it.get_group().get_linear_id();

        sycl::group<1> grp = it.get_group();
        const size_t in_offset = glb_idx << 1;

        scratch[loc_idx] = in_ptr[in_offset + 0] + in_ptr[in_offset + 1];

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
          out_itmd_ptr[wg_idx] = scratch[0];
        }
      });
  });

  // finally work-group local sums are added together (sequentially)
  // to find whole sum of set
  sycl::event evt_1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt_0);
    h.single_task<kernelSummationMethod2Phase1>([=]() {
      sycl::device_ptr<sycl::uint> out_itmd_ptr{ out_itmd };
      sycl::device_ptr<sycl::uint> out_final_ptr{ out_final };

      sycl::uint tmp = 0;
      for (size_t i = 0; i < out_itmd_len; i++) {
        tmp += out_itmd_ptr[i];
      }

      out_final_ptr[0] = tmp;
    });
  });

  return { evt_0, evt_1 };
}
}
