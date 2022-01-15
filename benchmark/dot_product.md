# Benchmarking Dot Product kernels targeting **Intel Stratix 10**

Each of kernel variants, computing dot product, takes two input vectors each of length **2 ^ 24**, where each element is 32-bit unsigned integer ( i.e. `sycl::uint` ). After computing dot of two input vectors on FPGA, output is asserted against result of one sequential function, executed on CPU --- just to be sure.

## Submit H/W Compilation Job

```bash
#!/bin/bash

# file name: build_fpga_hw.sh

# setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware compilation
TARGET_KERNEL=dot_product_method_0 make fpga_hw_test
```

```bash
# shell command to submit fpga compile job on Intel Devcloud,
# issued from login node

qsub -l nodes=1:fpga_compile:ppn=2 -l walltime=12:00:00 -d . build_fpga_hw.sh

# obtain job id here, looks like `1837254.v-qsvr-1.aidevcloud`
# job id is then `1837254`, to be used in next step
```

---

## Submit H/W Image Execution Job

```bash
#!/bin/bash

# file name: run_fpga_hw.sh

# setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware image execution
pushd test; ./fpga_hw.out; popd
```

```bash
# shell command to execute fpga hardware image on Intel Devcloud,
# issued from login node

qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . run_fpga_hw.sh -W depend=afterok:<job-id>

# fill in job id, obtained after enqueuing fpga compilation job
```

## Results

### dot_product_method_0

```bash
running on pac_s10 : Intel PAC Platform (pac_ee00000)

passed dot_product ( method_0 ) test            in 5038.65 ms
passed dot_product ( method_0 ) test            in 5038.65 ms
passed dot_product ( method_0 ) test            in 5038.66 ms
passed dot_product ( method_0 ) test            in 5038.65 ms
passed dot_product ( method_0 ) test            in 5038.66 ms
passed dot_product ( method_0 ) test            in 5038.65 ms
passed dot_product ( method_0 ) test            in 5038.66 ms
passed dot_product ( method_0 ) test            in 5038.65 ms

avg 5038.65 ms
```

### dot_product_method_2

```bash
running on pac_s10 : Intel PAC Platform (pac_ee00000)

passed dot_product ( method_2 ) test            in 5.44901 ms
passed dot_product ( method_2 ) test            in 5.44367 ms
passed dot_product ( method_2 ) test            in 5.44328 ms
passed dot_product ( method_2 ) test            in 5.44322 ms
passed dot_product ( method_2 ) test            in 5.4446 ms
passed dot_product ( method_2 ) test            in 5.44467 ms
passed dot_product ( method_2 ) test            in 5.44393 ms
passed dot_product ( method_2 ) test            in 5.44411 ms

avg 5.44456 ms
```

### dot_product_method_3

```bash
running on pac_s10 : Intel PAC Platform (pac_ee00000)

passed dot_product ( method_3 ) test            in 90.9964 ms
passed dot_product ( method_3 ) test            in 90.9959 ms
passed dot_product ( method_3 ) test            in 90.995 ms
passed dot_product ( method_3 ) test            in 90.9958 ms
passed dot_product ( method_3 ) test            in 90.9953 ms
passed dot_product ( method_3 ) test            in 90.9966 ms
passed dot_product ( method_3 ) test            in 90.9956 ms
passed dot_product ( method_3 ) test            in 90.9965 ms

avg 90.9959 ms
```
