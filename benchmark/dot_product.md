# Benchmarking Dot Product kernels targeting **Intel Stratix 10**

Each of kernel variants, computing dot product, takes two input vectors, each of length **2 ^ 24**, where each element is 32-bit unsigned integer ( i.e. `sycl::uint` ). After computing dot product of two input vectors on FPGA, output is asserted against result of one sequential function, executed on CPU --- just to be sure.

All these benchmarks are run on Intel Devcloud platform. Below I supply some scripts which I used for interacting with job submission queue. Two kinds of job submissions are presented below

- FPGA h/w compilation phase
- FPGA h/w image execution phase

## Submit H/W Compilation Job

Create a bash script for submitting compilation job on `fpga_compile` node.

```bash
touch build_fpga_hw.sh
```

And populate it with following content.

```bash
#!/bin/bash

# file name: build_fpga_hw.sh

# env setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware compilation of one kernel ( choose which one )
TARGET_KERNEL=dot_product_method_{0,1,2,3} make fpga_hw_test
```

Then execute following shell command to submit job to queue, with 12 hours timeout. Note the job id you get, it'll be used when submitting fpga image execution job, to form a dependency chain.

```bash
# shell command to submit fpga compile job on Intel Devcloud,
# issued from login node

qsub -l nodes=1:fpga_compile:ppn=2 -l walltime=12:00:00 -d . build_fpga_hw.sh

# obtain job id here, looks like `1837254.v-qsvr-1.aidevcloud`
# job id is then `1837254`, to be used in next step
```

---

## Submit H/W Image Execution Job

Create another shell script

```bash
touch run_fpga_hw.sh
```

Which will be populated with below content to execute fpga h/w image generated in previous step.

```bash
#!/bin/bash

# file name: run_fpga_hw.sh

# env setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware image execution
pushd test; ./fpga_hw.out; popd
```

Finally issue following shell command to submit h/w execution job with correct compilation job id, so that formed execution dependency chain is correct. This means once compilation job is completed on `fpga_compile` node on Devcloud, this task will start and collect benchmark results, while running h/w image on `fpga_runtime` node.

```bash
# shell command to execute fpga hardware image on Intel Devcloud,
# issued from login node

qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . run_fpga_hw.sh -W depend=afterok:<job-id>

# fill in job id, obtained after enqueuing fpga compilation job
```

> Note, h/w image execution target is set to `stratix10`, because h/w compilation phase also targeted that fpga. **You may want to read [this](https://github.com/itzmeanjan/fpga-explore/blob/849c728bc9b514fa60183f45b2f58328ece3bd31/Makefile#L11-L21).**

## Results

### dot_product_method_0

```bash
running on pac_s10 : Intel PAC Platform (pac_ec00000)

passed dot_product ( method_0 ) test            in 5236.66 ms
passed dot_product ( method_0 ) test            in 5236.64 ms
passed dot_product ( method_0 ) test            in 5236.66 ms
passed dot_product ( method_0 ) test            in 5236.66 ms
passed dot_product ( method_0 ) test            in 5236.64 ms
passed dot_product ( method_0 ) test            in 5236.66 ms
passed dot_product ( method_0 ) test            in 5236.64 ms
passed dot_product ( method_0 ) test            in 5236.65 ms

avg 5236.65 ms
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

### dot_product_method_4

```bash
running on pac_s10 : Intel PAC Platform (pac_ee00000)

passed dot_product ( method_4 ) test            in 273.292 ms
passed dot_product ( method_4 ) test            in 273.289 ms
passed dot_product ( method_4 ) test            in 273.291 ms
passed dot_product ( method_4 ) test            in 273.29 ms
passed dot_product ( method_4 ) test            in 273.292 ms
passed dot_product ( method_4 ) test            in 273.289 ms
passed dot_product ( method_4 ) test            in 273.291 ms
passed dot_product ( method_4 ) test            in 273.292 ms

avg 273.291 ms
```
