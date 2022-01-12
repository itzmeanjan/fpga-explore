#pragma once
#include "summation.hpp"
#include "utils.hpp"
#include <random>

namespace test_summation {
void
random_fill(sycl::uint* const in, size_t in_len)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<sycl::uint> dis(1u, 0xffffffffu);

  for (size_t i = 0; i < in_len; i++) {
    *(in + i) = dis(gen);
  }
}

void
seq_sum(const sycl::uint* in, size_t in_len, sycl::uint* const out)
{
  sycl::uint tmp = 0;
  for (size_t i = 0; i < in_len; i++) {
    tmp += *(in + i);
  }
  *out = tmp;
}

sycl::cl_ulong
method_0(sycl::queue& q, size_t in_len, size_t wg_size)
{
  sycl::uint* in_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint) * in_len, q));
  sycl::uint* in_d = static_cast<sycl::uint*>(
    sycl::malloc_device(sizeof(sycl::uint) * in_len, q));
  sycl::uint* out_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint), q));
  sycl::uint* out_d =
    static_cast<sycl::uint*>(sycl::malloc_device(sizeof(sycl::uint), q));

  random_fill(in_h, in_len);

  sycl::event evt_0 = q.memcpy(in_d, in_h, sizeof(sycl::uint) * in_len);
  sycl::event evt_1 = q.memset(out_d, 0, sizeof(sycl::uint));
  sycl::event evt_2 =
    summation::method_0(q, in_d, in_len, out_d, wg_size, { evt_0, evt_1 });
  sycl::event evt_3 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt_2);
    h.memcpy(out_h, out_d, sizeof(sycl::uint));
  });

  evt_3.wait();

  // device computed result to be compared against this
  sycl::uint out_cmp = 0;
  seq_sum(in_h, in_len, &out_cmp);

  assert(*out_h == out_cmp);

  sycl::cl_ulong ts = time_event(evt_2);

  sycl::free(in_h, q);
  sycl::free(in_d, q);
  sycl::free(out_h, q);
  sycl::free(out_d, q);

  return ts;
}

sycl::cl_ulong
method_1(sycl::queue& q, size_t in_len, size_t wg_size)
{
  sycl::uint* in_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint) * in_len, q));
  sycl::uint* in_d = static_cast<sycl::uint*>(
    sycl::malloc_device(sizeof(sycl::uint) * in_len, q));
  sycl::uint* out_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint), q));
  sycl::uint* out_d =
    static_cast<sycl::uint*>(sycl::malloc_device(sizeof(sycl::uint), q));

  random_fill(in_h, in_len);

  sycl::event evt_0 = q.memcpy(in_d, in_h, sizeof(sycl::uint) * in_len);
  sycl::event evt_1 = q.memset(out_d, 0, sizeof(sycl::uint));
  sycl::event evt_2 =
    summation::method_1(q, in_d, in_len, out_d, wg_size, { evt_0, evt_1 });
  sycl::event evt_3 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt_2);
    h.memcpy(out_h, out_d, sizeof(sycl::uint));
  });

  evt_3.wait();

  // device computed result to be compared against this
  sycl::uint out_cmp = 0;
  seq_sum(in_h, in_len, &out_cmp);

  assert(*out_h == out_cmp);

  sycl::cl_ulong ts = time_event(evt_2);

  sycl::free(in_h, q);
  sycl::free(in_d, q);
  sycl::free(out_h, q);
  sycl::free(out_d, q);

  return ts;
}

sycl::cl_ulong
method_2(sycl::queue& q, size_t in_len, size_t wg_size)
{
  const size_t total_work_items = in_len >> 1;
  const size_t rev_wg_size =
    wg_size <= total_work_items ? wg_size : total_work_items;
  const size_t out_len = total_work_items / rev_wg_size;

  sycl::uint* in_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint) * in_len, q));
  sycl::uint* in_d = static_cast<sycl::uint*>(
    sycl::malloc_device(sizeof(sycl::uint) * in_len, q));
  sycl::uint* out_h = static_cast<sycl::uint*>(
    sycl::malloc_host(sizeof(sycl::uint) * out_len, q));
  sycl::uint* out_d = static_cast<sycl::uint*>(
    sycl::malloc_device(sizeof(sycl::uint) * out_len, q));
  sycl::uint* out_final_h =
    static_cast<sycl::uint*>(sycl::malloc_host(sizeof(sycl::uint), q));
  sycl::uint* out_final_d =
    static_cast<sycl::uint*>(sycl::malloc_device(sizeof(sycl::uint), q));

  random_fill(in_h, in_len);

  sycl::event evt_0 = q.memcpy(in_d, in_h, sizeof(sycl::uint) * in_len);
  std::vector<sycl::event> evts = summation::method_2(
    q, in_d, in_len, out_d, out_len, out_final_d, wg_size, { evt_0 });
  sycl::event evt_3 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts.at(1));
    h.memcpy(out_final_h, out_final_d, sizeof(sycl::uint));
  });

  evt_3.wait();

  // device computed result to be compared against this
  // host computed result
  sycl::uint out_cmp = 0;
  seq_sum(in_h, in_len, &out_cmp);

  assert(*out_final_h == out_cmp);

  sycl::cl_ulong ts = time_event(evts.at(0)) + time_event(evts.at(1));

  sycl::free(in_h, q);
  sycl::free(in_d, q);
  sycl::free(out_h, q);
  sycl::free(out_d, q);

  return ts;
}
}
