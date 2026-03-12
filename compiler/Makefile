CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -MMD -MP
LDFLAGS := -lprotobuf

SRC_DIR := .
BUILD_DIR := build

TARGET := compiler.elf
TARGET_ARGS ?=

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

.PHONY: all run clean format

all: $(TARGET)
	@echo "$(COLOR_GREEN)[SUCCESS]$(COLOR_RESET) $(TARGET) built successfully."

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

clean:
	@echo "$(COLOR_YELLOW)[CLEAN]$(COLOR_RESET) Removing build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(TARGET)

-include $(DEPS)
