#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace summation {
std::vector<sycl::event>
method_3(sycl::queue& q,
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
  assert(in_len == out_len);

  const size_t total_work_items = in_len >> 1;
  const size_t rev_wg_size =
    wg_size <= total_work_items ? wg_size : total_work_items;
  const size_t out_offset = total_work_items;

  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.parallel_for<class kernelSummationMethod3Phase0>(
      sycl::nd_range<1>{ sycl::range<1>{ total_work_items },
                         sycl::range<1>{ rev_wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();
        const size_t in_offset = idx << 1;

        *(out + out_offset + idx) = *(in + in_offset) + *(in + in_offset + 1);
      });
  });

  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(total_work_items)));

  std::vector<sycl::event> _evts;
  _evts.reserve(rounds + 1);

  _evts.push_back(evt_0);

  if (rounds > 0) {
    for (size_t r = 0; r < rounds; r++) {
      sycl::event evt = q.submit([&](sycl::handler& h) {
        h.depends_on(_evts.at(r));

        const size_t req_wi_cnt = total_work_items >> (r + 1);
        const size_t rev_wg_size = wg_size <= req_wi_cnt ? wg_size : req_wi_cnt;
        const size_t out_offset = req_wi_cnt;

        h.parallel_for<class kernelSummationMethod3Phase1>(
          sycl::nd_range<1>{ sycl::range<1>{ req_wi_cnt },
                             sycl::range<1>{ rev_wg_size } },
          [=](sycl::nd_item<1> it) {
            const size_t idx = it.get_global_linear_id();
            const size_t in_offset = (out_offset << 1) + (idx << 1);

            *(out + out_offset + idx) =
              *(out + in_offset) + *(out + in_offset + 1);
          });
      });

      _evts.push_back(evt);
    }
  }

  return _evts;
}
}
