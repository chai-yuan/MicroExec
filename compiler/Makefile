CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -MMD -MP -DUSE_ONNX_SHAPE_INFERENCE -DONNX_ML -DONNX_NAMESPACE=onnx
LDFLAGS := -lprotobuf -lonnx -lonnx_proto

SRC_DIR := .
BUILD_DIR := build
FRONTEND_DIR := $(SRC_DIR)/frontend

TARGET := compiler.elf
TARGET_ARGS ?=

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cc) \
        $(wildcard $(SRC_DIR)/frontend/*.cc) \
        $(wildcard $(SRC_DIR)/graph/*.cc) \
        $(wildcard $(SRC_DIR)/execution/*.cc) \
        $(wildcard $(SRC_DIR)/backend/*.cc)

HEADERS := $(wildcard $(SRC_DIR)/*.h) \
           $(wildcard $(SRC_DIR)/frontend/*.h) \
           $(wildcard $(SRC_DIR)/graph/*.h) \
           $(wildcard $(SRC_DIR)/execution/*.h) \
           $(wildcard $(SRC_DIR)/backend/*.h)

OBJS := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

INCLUDES := -I$(SRC_DIR)

COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m
COLOR_CYAN := \033[36m

.PHONY: all init run clean format clean-all

# Default target: build the compiler
all: $(TARGET)

run: all
	@echo "$(COLOR_CYAN)[RUN]$(COLOR_RESET) Executing $(TARGET)..."
	./$(TARGET) $(TARGET_ARGS)

$(BUILD_DIR)/%.o: %.cc
	@echo "$(COLOR_YELLOW)[CXX]$(COLOR_RESET) $<..."
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	@echo "$(COLOR_BLUE)[LINK]$(COLOR_RESET) $@..."
	@$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)


# Protobuf files
PROTO_FILE := $(FRONTEND_DIR)/onnx.proto
PROTO_HDR := $(FRONTEND_DIR)/onnx.pb.h
PROTO_SRC := $(FRONTEND_DIR)/onnx.pb.cc

clean:
	@echo "$(COLOR_YELLOW)[CLEAN]$(COLOR_RESET) Removing build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(TARGET)

clean-all: clean
	@echo "$(COLOR_YELLOW)[CLEAN]$(COLOR_RESET) Removing generated protobuf files..."
	@rm -f $(PROTO_HDR) $(PROTO_SRC)

-include $(DEPS)
