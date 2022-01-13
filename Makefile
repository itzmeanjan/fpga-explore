CXX = dpcpp
CXX_FLAGS = -std=c++20 -Wall
SYCL_FLAGS = -fsycl
SYCL_CUDA_FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
INCLUDES = -I./include
SYCL_FPGA_EMU_FLAGS = -DTARGET_FPGA_EMU -fintelfpga
SYCL_FPGA_FLAGS = -DTARGET_FPGA -fintelfpga -fsycl-link=early -Xshardware

# I suggest seeing https://github.com/itzmeanjan/fpga-explore/blob/434e9ff/include/utils.hpp#L9-L29
# for understanding possibilities
#
# Possible values for TARGET can be
#
# - cpu
# - gpu
# - fpga_emu
# - fpga
# - default
#
# You probably want to specify it when invoking make utility as `$ TARGET=fpga_emu make`
TARGET_FLAGS = -DTARGET_$(shell echo $(or $(TARGET),default) | tr a-z A-Z)

# Possible values for TARGET_KERNEL 
#
# - summation_method_0
# - summation_method_1
# - summation_method_2
# - summation_method_3
# - summation_method_4
# - dot_product_method_0
#
# You want to specify it when invoking make as `$ TARGET_KERNEL=summation_method_0 make`
TARGET_KERNEL_FLAGS = -D$(shell echo $(or $(TARGET_KERNEL),place_holder))

all: test

test: test/a.out
	./$<

test/a.out: test/main.cpp include/*.hpp
	$(CXX) $(CXX_FLAGS) $(SYCL_FLAGS) $(TARGET_FLAGS) $(TARGET_KERNEL_FLAGS) $(INCLUDES) $< -o $@

fpga_emu_test:
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(TARGET_KERNEL_FLAGS) $(SYCL_FPGA_EMU_FLAGS) test/main.cpp -o test/a.out

fpga_test:
	@if [ $(TARGET_KERNEL_FLAGS) != '-Dplace_holder' ]; then \
		$(CXX) $(CXX_FLAGS) $(INCLUDES) $(TARGET_KERNEL_FLAGS) $(SYCL_FPGA_FLAGS)  test/main.cpp -o test/a.out; \
	else \
		echo "Must select kernel !"; \
	fi

clean:
	find . -name 'a.out' -o -name '*.o' -o -name '*.prj' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla
