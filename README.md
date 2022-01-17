# fpga-explore
Exploring FPGA design space using SYCL/ DPC++ HLS Tool

## Motivation

After exploring heterogeneous computing using SYCL for some time, while majorly targeting CPU & GPU, I planned to dive into FPGA world, taking help of High Level Synthesis tools offered by SYCL/ DPC++.

In this repository I keep multiple competing implementations of kernels which I've written/ will write over time, targeting FPGA. I keep multiple kernels solving same problem ( say reduction ) to show design choice evolution phases, I've come across, while writing these kernels. Along with that it also helps me in understanding & keeping track of which FPGA specific attributes/ features perform how for some specific problem. To be more specific when loop unrolling can be utilised, where single work-item kernel performs better than multi-work item kernel, which loop unroll factor should be chosen for specific problem ( read loop under considerataion ) or whether to use `fpga_registers` or not for representing small arrays etc.

At this moment, I'm keeping multiple implementations of following problems.

- summation ( read reduction )
- dot_product
- mat_mul ( matrix multiplication )

I follow header-only library style in this repository, where kernels are ( namespaced ) kept inside respective directory denoting problem name under `./include`. As there are multiple implementations of same problem statement, I make use of preprocessor directives to choose which kernel to include at compile time.

After h/w compilation phase, I execute these h/w images on **Intel Stratix 10 FPGA Board** ( on Intel Devcloud ) for seeing how performant are they. You can find benchmark results in [this](./benchmark) directory.

## Prerequisites

- I've Intel oneAPI basekit installed with `dpcpp` version

```bash
Intel(R) oneAPI DPC++/C++ Compiler 2022.0.0 (2022.0.0.20211123)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/intel/oneapi/compiler/2022.0.1/linux/bin-llvm
```

- For FPGA emultion & optimization report generation ( based on partial compilation ), I make use of host device, having no FPGA board attached to it.
- For h/w compilation & execution, I rely on Intel Devcloud platform.

> Note, necessary scripts for submitting h/w compilation & execution jobs to Devcloud job queues, are provided by files present in [benchmark](./benchmark) directory.

- You will also need to have `make` for issuing build commands.
- If you modify source code, consider running `make format` after that, to maintain same code formatting convention, which uses `clang-format`'s Mozilla format style.

## Usage

Assuming you've oneAPI basekit installed, you must also have FPGA Emulator, which can be targeted for checking functional correctness of various kernels.

```bash
$ sycl-ls | grep -i fpga

[opencl:0] ACC : Intel(R) FPGA Emulation Platform for OpenCL(TM) 1.2 [2021.13.11.0.23_160000]
```

Consider executing following for running all test cases on FPGA emulator.

```bash
chmod +x run.sh
./run.sh
```

You may want to generate optimization report for some kernel, which can be done only using `dpcpp`


```bash
make clean
TARGET_KERNEL=summation_method_0 make fpga_opt_test
make clean

ls test/fpga_opt/reports/report.html        # You're interested in !
<browser> test/fpga_opt/reports/report.html # Analyze fpga opt report
```

> I suggest you see [Makefile](Makefile) to understand possible options for selecting which kernel to compile.

## Benchmark

Following benchmark results along with how to submit benchmark compilation/ execution job, are provided

- [summation](benchmark/summation.md)
- [dot_product](benchmark/dot_product.md)
- [mat_mul](benchmark/mat_mul.md)
