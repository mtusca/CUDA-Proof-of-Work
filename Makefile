SRC_DIR ?= src
EXEC = miner

ifndef COMPILER
$(error COMPILER is not defined)
endif

ifndef LOCAL

# Partitions
BUILD_PARTITION    ?= xl
RUN_PARTITION      ?= $(BUILD_PARTITION)

# Time limits
RUN_TIME      ?= 00:01:00
BUILD_TIME    ?= 00:01:00

# Commands
BUILD_CMD    ?= make build LOCAL=y
RUN_CMD      ?= make run LOCAL=y
LOAD_PREFIX  ?= apptainer exec --env='TMPDIR=$(HOME)' --nv $(IMG_PATH)

# Image config
IMG_TAG  ?= 12.2.2
IMG_PATH ?= /export/home/acs/prof/stefan_dan.ciocirlan/TMP/DO_NOT_DELETE_IMGS/cuda-labs_$(IMG_TAG).sif

build: $(IMG_PATH)
	@sbatch --time $(BUILD_TIME) --partition $(BUILD_PARTITION) --wrap="$(LOAD_PREFIX) $(BUILD_CMD)"

run: $(IMG_PATH)
ifeq ($(COMPILER), gcc)
	@sbatch --time $(RUN_TIME) --partition $(RUN_PARTITION) --wrap="$(LOAD_PREFIX) $(RUN_CMD)"
else ifeq ($(COMPILER), nvcc)
	@sbatch --gres gpu:1 --time $(RUN_TIME) --partition $(RUN_PARTITION) --wrap="$(LOAD_PREFIX) $(RUN_CMD)"
endif

else

COMMON_DIR ?= ../common
OBJS ?= $(SRC_DIR)/miner.o $(SRC_DIR)/sha256.o $(SRC_DIR)/utils.o
LIBS ?= -lm
ifeq ($(COMPILER), gcc)
FORCE_C := -x c
else
FORCE_C :=
endif

build: $(EXEC)

$(EXEC): $(OBJS)
	$(COMPILER) $(OBJS) -o $(EXEC) $(LIBS)

$(SRC_DIR)/miner.o: $(COMMON_DIR)/miner.cpp
	$(COMPILER) $(CFLAGS) $(FORCE_C) -c $< -o $@

$(SRC_DIR)/sha256.o: $(SRC_DIR)/sha256.$(SRC_EXT)
	$(COMPILER) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/utils.o: $(SRC_DIR)/utils.$(SRC_EXT)
	$(COMPILER) $(CFLAGS) -c $< -o $@

run: $(EXEC)
	@echo "Running test $(TEST)..."
	./$(EXEC) $(TEST)

endif

clean:
	rm -f $(SRC_DIR)/*.o output/* $(EXEC) slurm-*.out slurm-*.err profile.ncu-rep

.PHONY: build run clean
