# Benchmarking Matrix Multiplication kernels targeting **Intel Stratix 10**

Each kernel computing matrix multiplication takes two input matrices of dimension (1024 x 2048) & (2048 x 4096), producing output matrix of dimension (1024 x 4096), while using local work group size (1, 32, 1). Before finalising benchmark result, it executes kernel 8 times and takes average. Finally just to be sure that mat_mul result computed by kernel is correct, I take help of one O(n^2) (sequentially executing) host program to assert each cell of output matrix.

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
TARGET_KERNEL=mat_mul_method_{0} make fpga_hw_test
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

Finally issue following shell command to submit h/w execution job with correct compilation phase job id, so that correct execution dependency chain is formed. This means once compilation job is completed on `fpga_compile` node on Devcloud, this task will start and collect benchmark results, while running h/w image on `fpga_runtime` node.

```bash
# shell command to execute fpga hardware image on Intel Devcloud,
# issued from login node

qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . run_fpga_hw.sh -W depend=afterok:<job-id>

# fill in job id, obtained after enqueuing fpga compilation job
```

> Note, h/w image execution target is set to `stratix10`, because h/w compilation phase also targeted that fpga. **You may want to read [this](https://github.com/itzmeanjan/fpga-explore/blob/849c728bc9b514fa60183f45b2f58328ece3bd31/Makefile#L11-L21).**

## Results

### mat_mul_method_0

```bash
running on pac_s10 : Intel PAC Platform (pac_ec00000)

passed mat_mul ( method_0 ) test                in 3315.63 ms
passed mat_mul ( method_0 ) test                in 3229.04 ms
passed mat_mul ( method_0 ) test                in 3198.42 ms
passed mat_mul ( method_0 ) test                in 3218.26 ms
passed mat_mul ( method_0 ) test                in 3251.73 ms
passed mat_mul ( method_0 ) test                in 3325.72 ms
passed mat_mul ( method_0 ) test                in 3322.07 ms
passed mat_mul ( method_0 ) test                in 3344.09 ms

avg 3275.62 ms
```