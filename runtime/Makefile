CC      := gcc
AR      := ar
CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -MMD -MP -O2
ARFLAGS := rcs

INCLUDES := -Iinclude -Ioperators -Isrc -I../compiler/common

SRC_DIR  := src
OPS_DIR  := operators
BUILD_DIR := build

LIB_TARGET := libmicroexec.a

SRCS := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(OPS_DIR)/*.c)
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

COLOR_RESET  := \033[0m
COLOR_GREEN  := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE   := \033[34m

.PHONY: all clean examples

all: $(LIB_TARGET)
	@echo "$(COLOR_GREEN)[SUCCESS]$(COLOR_RESET) $(LIB_TARGET) built successfully."

$(LIB_TARGET): $(OBJS)
	@echo "$(COLOR_BLUE)[AR]$(COLOR_RESET) $@..."
	@$(AR) $(ARFLAGS) $@ $^

$(BUILD_DIR)/%.o: %.c
	@echo "$(COLOR_YELLOW)[CC]$(COLOR_RESET) $<..."
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

examples: $(LIB_TARGET)
	@$(MAKE) -C examples/01_basic_inference
	@$(MAKE) -C examples/02_custom_operator
	@$(MAKE) -C examples/03_x86_simd

clean:
	@echo "$(COLOR_YELLOW)[CLEAN]$(COLOR_RESET) Removing build artifacts..."
	@rm -rf $(BUILD_DIR) $(LIB_TARGET)
	@$(MAKE) -C examples/01_basic_inference clean 2>/dev/null || true
	@$(MAKE) -C examples/02_custom_operator clean 2>/dev/null || true
	@$(MAKE) -C examples/03_x86_simd clean 2>/dev/null || true

-include $(DEPS)
