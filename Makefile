JOBS?=$(bash getconf _NPROCESSORS_CONF)

.PHONY: clean build

clean:
	@ls | grep '^build-\(Debug\|Release\)' | xargs -r rm -r

build:
	@mkdir -p build-$(BUILD_TYPE) && \
	cmake -B build-$(BUILD_TYPE) -GNinja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) 
	cmake --build build-$(BUILD_TYPE) -j $(JOBS) 

debug:
	$(MAKE) build BUILD_TYPE=Debug ENABLE_SANITIZER=OFF

debug-asn:
	$(MAKE) build BUILD_TYPE=Debug ENABLE_SANITIZER=ON

release:
	$(MAKE) build BUILD_TYPE=Release ENABLE_SANITIZER=OFF

release-debug:
	$(MAKE) build BUILD_TYPE=RelWithDebInfo ENABLE_SANITIZER=OFF

sanitizer:
	echo 1 > /proc/sys/vm/overcommit_memory

py_install:
	MAX_JOBS=JOBS pip install --no-build-isolation  -v -e .

HF_TP_bench:
	cd benchmarks && python benchmark_throughput.py --backend=hf --dataset=../ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/vllm/vicuna-7b-v1.5/ --n=1 --num-prompts=350 --hf-max-batch-size=10 --trust-remote-code --cpu-only

VLLM_TP_bench:
	cd benchmarks && python benchmark_throughput.py --backend=vllm --dataset=../ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/vllm/vicuna-7b-v1.5/ --n=1 --num-prompts=1 --trust-remote-code --cpu-only --swap-space=1

VLLM_LT_bench:
	cd benchmarks && python benchmark_latency.py --model=/root/frameworks.bigdata.dev-ops/vicuna-7b-v1.5/ --n=1 --batch-size=1 --num-iters=1000 --trust-remote-code --cpu-only 

HF_TP_bench_gpu:
	cd benchmarks && python benchmark_throughput.py --backend=hf --dataset=../ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/vllm/vicuna-7b-v1.5/ --n=1 --num-prompts=1000 --hf-max-batch-size=8 --trust-remote-code

VLLM_TP_bench_gpu:
	cd benchmarks && python benchmark_throughput.py --backend=vllm --dataset=../ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/vllm/vicuna-7b-v1.5/ --n=1 --num-prompts=1 --trust-remote-code