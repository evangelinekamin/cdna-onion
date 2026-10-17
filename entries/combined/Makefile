# onion-walk: Tor v3 vanity grinder for AMD CDNA (onionrig-grind/1).
#   make entry GPU_ARCH=gfx90a   the arena's build: ./entry (several archs: one fat binary)
#   make test                    CPU tests (reference math, field code, walk emulation)
#   make resource-usage          VGPRs / SGPRs / scratch / occupancy per kernel
#   make isa                     AMD assembly, for reading the hot loop
# Tuning knobs are preprocessor defines, e.g. make entry DEFS="-DWALK_H=2048".

HIPCC     ?= hipcc
CXX       ?= g++
GPU_ARCH  ?= gfx90a
DEFS      ?=
SRC        = src/grind.hip
DEPS       = $(wildcard src/*.h src/*.hip)
HIPFLAGS   = -O3 -std=c++17 -fno-slp-vectorize $(DEFS)
OFFLOAD    = $(addprefix --offload-arch=,$(GPU_ARCH))
LDLIBS     = -lpthread

.PHONY: all test resource-usage isa clean
all: entry

entry: $(DEPS)
	$(HIPCC) $(HIPFLAGS) $(OFFLOAD) -o $@ $(SRC) $(LDLIBS)

build:
	mkdir -p build

build/%: test/%.cpp $(DEPS) | build
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas $(DEFS) -o $@ $<

test: build/ref_test build/fe29_test build/walk_test
	./build/ref_test ../../vectors
	./build/fe29_test
	./build/walk_test

resource-usage: | build
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(GPU_ARCH) --cuda-device-only -c $(SRC) \
	  -o build/grind-$(GPU_ARCH).o -Rpass-analysis=kernel-resource-usage 2> build/resources-$(GPU_ARCH).txt
	@grep -E "Function Name|VGPRs:|AGPRs|ScratchSize|Occupancy|SGPRs:|Spill" build/resources-$(GPU_ARCH).txt \
	  | sed -E 's/.*remark: //; s/ \[-Rpass-analysis=kernel-resource-usage\]//'

isa: | build
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(GPU_ARCH) --cuda-device-only -S $(SRC) -o build/grind-$(GPU_ARCH).s
	@echo "wrote build/grind-$(GPU_ARCH).s"

clean:
	rm -rf build entry
