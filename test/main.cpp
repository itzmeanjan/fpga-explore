#include "test_summation.hpp"
#include "utils.hpp"
#include <iostream>

#if defined(summation_method_0) || defined(summation_method_1) ||              \
  defined(summation_method_2) || defined(summation_method_3) ||                \
  defined(summation_method_4)
constexpr size_t IN_LEN = 1 << 24;
constexpr size_t WG_SIZE = 1 << 5;
#endif

int
main(int argc, char** argv)
{
  void* wq;
  make_queue(&wq);
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  std::cout << "running on "
            << q->get_device().get_info<sycl::info::device::name>() << std::endl
            << std::endl;

#if defined(summation_method_0)

  sycl::cl_ulong ts = test_summation::method_0(*q, IN_LEN, WG_SIZE);
  std::cout << "passed summation ( method_0 ) test\t\t"
            << "in " << (double)ts * 1e-6 << " ms" << std::endl;

#elif defined(summation_method_1)

  sycl::cl_ulong ts = test_summation::method_1(*q, IN_LEN, WG_SIZE);
  std::cout << "passed summation ( method_1 ) test\t\t"
            << "in " << (double)ts * 1e-6 << " ms" << std::endl;

#elif defined(summation_method_2)

  sycl::cl_ulong ts = test_summation::method_2(*q, IN_LEN, WG_SIZE);
  std::cout << "passed summation ( method_1 ) test\t\t"
            << "in " << (double)ts * 1e-6 << " ms" << std::endl;

#elif defined(summation_method_3)

  sycl::cl_ulong ts = test_summation::method_3(*q, IN_LEN, WG_SIZE);
  std::cout << "passed summation ( method_1 ) test\t\t"
            << "in " << (double)ts * 1e-6 << " ms" << std::endl;

#elif defined(summation_method_4)

  sycl::cl_ulong ts = test_summation::method_4(*q, IN_LEN, WG_SIZE);
  std::cout << "passed summation ( method_1 ) test\t\t"
            << "in " << (double)ts * 1e-6 << " ms" << std::endl;

#else
  std::cout << "No kernel(s) to run !" << std::endl;
#endif

  return 0;
}
