# baseline-onion-gpu: one source, two backends.
#   make cuda            NVIDIA build for local testing (CUDA_ARCH=native by default)
#   make hip             AMD build (HIP_ARCHS="gfx90a gfx950" by default)
#   make resource-usage  per-kernel VGPRs / scratch / occupancy for each AMD arch
#   make isa             AMD assembly for each arch, for reading the hot loop
# Tuning knobs are preprocessor defines, e.g. make cuda DEFS="-DBINV_N=256".

NVCC      ?= $(firstword $(wildcard /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc) nvcc)
HIPCC     ?= hipcc
CUDA_ARCH ?= native
HIP_ARCHS ?= gfx90a gfx950
DEFS      ?=
SRC        = src/grind.cu
DEPS       = $(wildcard src/*.cu src/*.cuh src/*.h)
COMMON     = -O3 -std=c++17 $(DEFS)

.PHONY: all cuda hip resource-usage isa clean
all: cuda

cuda: build/grind-cuda
hip: build/grind-hip

build/grind-cuda: $(DEPS) | build
	$(NVCC) $(COMMON) -arch=$(CUDA_ARCH) -o $@ $(SRC)

build/grind-hip: $(DEPS) | build
	$(HIPCC) $(COMMON) $(addprefix --offload-arch=,$(HIP_ARCHS)) -x hip -o $@ $(SRC)

resource-usage: | build
	@for a in $(HIP_ARCHS); do \
	  $(HIPCC) $(COMMON) --offload-arch=$$a --cuda-device-only -x hip -c $(SRC) \
	    -o build/grind-$$a.o -Rpass-analysis=kernel-resource-usage 2> build/resources-$$a.txt; \
	  echo "== $$a"; grep -A12 "Function Name: .*grind_kernel" build/resources-$$a.txt \
	    | sed -E 's/.*remark: //; s/ \[-Rpass-analysis=kernel-resource-usage\]//'; \
	done

isa: | build
	@for a in $(HIP_ARCHS); do \
	  $(HIPCC) $(COMMON) --offload-arch=$$a --cuda-device-only -x hip -S $(SRC) \
	    -o build/grind-$$a.s && echo "wrote build/grind-$$a.s"; \
	done

build:
	mkdir -p build

clean:
	rm -rf build
