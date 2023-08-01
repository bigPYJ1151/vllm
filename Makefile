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