#include "test_summation.hpp"
#include "utils.hpp"
#include <iostream>

const size_t IN_LEN = 1 << 20;
const size_t WG_SIZE = 1 << 5;

int
main(int argc, char** argv)
{
  void* wq;
  make_queue(&wq);
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  std::cout << "running on "
            << q->get_device().get_info<sycl::info::device::name>()
            << std::endl;

  test_summation::method_0(*q, IN_LEN, WG_SIZE);
  std::cout << "passed summation ( method_0 ) test" << std::endl;

  return 0;
}
