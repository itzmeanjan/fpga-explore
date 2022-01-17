#!/bin/bash

# just to be safe !
make clean

# summation
TARGET_KERNEL=summation_method_0 make fpga_emu_test; make clean
TARGET_KERNEL=summation_method_1 make fpga_emu_test; make clean
TARGET_KERNEL=summation_method_2 make fpga_emu_test; make clean
TARGET_KERNEL=summation_method_3 make fpga_emu_test; make clean
TARGET_KERNEL=summation_method_4 make fpga_emu_test; make clean
TARGET_KERNEL=summation_method_5 make fpga_emu_test; make clean

# dot product
TARGET_KERNEL=dot_product_method_0 make fpga_emu_test; make clean
TARGET_KERNEL=dot_product_method_1 make fpga_emu_test; make clean
TARGET_KERNEL=dot_product_method_2 make fpga_emu_test; make clean
TARGET_KERNEL=dot_product_method_3 make fpga_emu_test; make clean

# matrix multiplication
TARGET_KERNEL=mat_mul_method_0 make fpga_emu_test; make clean
