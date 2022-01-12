CXX = dpcpp
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
INCLUDES = -I./include

all: test

test: test/a.out
	./$<

test/a.out: test/main.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) $< -o $@

clean:
	find . -name 'a.out' -o -name '*.o' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla