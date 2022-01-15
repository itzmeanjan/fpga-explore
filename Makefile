CXX = dpcpp
CXX_FLAGS = -std=c++20 -Wall
SYCL_FLAGS = -fsycl
SYCL_CUDA_FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
INCLUDES = -I./include
SYCL_FPGA_EMU_FLAGS = -DTARGET_FPGA_EMU -fintelfpga

# You can change target board to `intel_a10gx_pac:pac_a10`
SYCL_FPGA_OPT_FLAGS = -DTARGET_FPGA -fintelfpga -fsycl-link=early -Xshardware -Xsboard=intel_s10sx_pac:pac_s10

# During compilation if target board is set to `intel_s10sx_pac:pac_s10`
# ensure during runtime, it's executed on `fpga_runtime:stratix10` node on Intel devcloud
#
# or else you can consider setting target board to `intel_a10gx_pac:pac_a10`
# then make sure you can run that fpga hardware image on `fpga_runtime:arria10` node on
# Intel devcloud
#
# **Note**
# for now I'm not enabling `-Xsprofile`, which helps in profiling executable binary and
# collecting more data which can be further analysed using Intel `vtune`
SYCL_FPGA_HW_FLAGS = -DTARGET_FPGA -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10

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
# - dot_product_method_1
# - dot_product_method_2
# - dot_product_method_3
#
# You want to specify it when invoking make as `$ TARGET_KERNEL=summation_method_0 make`
TARGET_KERNEL_FLAGS = -D$(shell echo $(or $(TARGET_KERNEL),place_holder))

all: test

test: test/a.out
	./$<

test/a.out: test/main.cpp include/*.hpp
	$(CXX) $(CXX_FLAGS) $(SYCL_FLAGS) $(TARGET_FLAGS) $(TARGET_KERNEL_FLAGS) $(INCLUDES) $< -o $@

fpga_emu_test:
	@if [ $(TARGET_KERNEL_FLAGS) != '-Dplace_holder' ]; then \
		$(CXX) $(CXX_FLAGS) $(INCLUDES) $(TARGET_KERNEL_FLAGS) $(SYCL_FPGA_EMU_FLAGS)  test/main.cpp -o test/fpga_emu.out; \
		./test/fpga_emu.out; \
	else \
		echo "Must select kernel !"; \
	fi

fpga_opt_test:
	# output not supposed to be executed, instead consume report generated
	# inside `test/fpga_opt.prj/reports/` diretory
	@if [ $(TARGET_KERNEL_FLAGS) != '-Dplace_holder' ]; then \
		$(CXX) $(CXX_FLAGS) $(INCLUDES) $(TARGET_KERNEL_FLAGS) $(SYCL_FPGA_OPT_FLAGS)  test/main.cpp -o test/fpga_opt.a; \
	else \
		echo "Must select kernel !"; \
	fi

fpga_hw_test:
	@if [ $(TARGET_KERNEL_FLAGS) != '-Dplace_holder' ]; then \
		$(CXX) $(CXX_FLAGS) $(INCLUDES) $(TARGET_KERNEL_FLAGS) $(SYCL_FPGA_HW_FLAGS) -reuse-exe=test/fpga_hw.out test/main.cpp -o test/fpga_hw.out; \
	else \
		echo "Must select kernel !"; \
	fi

clean:
	find . -name '*.out' -o -name '*.o' -o -name '*.prj' -o -name '*.a' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla
