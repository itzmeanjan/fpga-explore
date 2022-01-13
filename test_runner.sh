#!/bin/bash

# just to be safe !
make clean

# summation
TARGET=cpu TARGET_KERNEL=summation_method_0 make; make clean
TARGET=cpu TARGET_KERNEL=summation_method_1 make; make clean
TARGET=cpu TARGET_KERNEL=summation_method_2 make; make clean
TARGET=cpu TARGET_KERNEL=summation_method_3 make; make clean
TARGET=cpu TARGET_KERNEL=summation_method_4 make; make clean

# dot product
TARGET=cpu TARGET_KERNEL=dot_product_method_0 make; make clean
TARGET=cpu TARGET_KERNEL=dot_product_method_1 make; make clean
TARGET=cpu TARGET_KERNEL=dot_product_method_2 make; make clean
