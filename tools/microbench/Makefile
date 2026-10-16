# Primitive-throughput microbenchmarks (see microbench.cu).
#   make cuda | make hip | make isa (checks each kernel uses the instruction it claims)

NVCC      ?= $(firstword $(wildcard /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc) nvcc)
HIPCC     ?= hipcc
CUDA_ARCH ?= native
HIP_ARCHS ?= gfx90a gfx950
FLAGS      = -O3 -std=c++17

.PHONY: cuda hip isa clean
cuda: build/microbench-cuda
hip: build/microbench-hip

build/microbench-cuda: microbench.cu | build
	$(NVCC) $(FLAGS) -arch=$(CUDA_ARCH) -o $@ $<

build/microbench-hip: microbench.cu | build
	$(HIPCC) $(FLAGS) $(addprefix --offload-arch=,$(HIP_ARCHS)) -x hip -o $@ $<

isa: | build
	@for a in $(HIP_ARCHS); do \
	  $(HIPCC) $(FLAGS) --offload-arch=$$a --cuda-device-only -x hip -S microbench.cu \
	    -o build/microbench-$$a.s && echo "wrote build/microbench-$$a.s"; \
	done

build:
	mkdir -p build

clean:
	rm -rf build
