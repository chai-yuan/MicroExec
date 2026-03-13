CC      := gcc
CFLAGS  := -std=c11 -Wall -Wextra -O2

RUNTIME_DIR := ../..
RUNTIME_LIB := $(RUNTIME_DIR)/libmicroexec.a
INCLUDES    := -I$(RUNTIME_DIR)/include

TARGET := basic_inference.elf

SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)

.PHONY: all clean lib

all: lib $(TARGET)
	@echo "[OK] $(TARGET)"

lib:
	@$(MAKE) -C $(RUNTIME_DIR) --no-print-directory

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ $(RUNTIME_LIB) -o $@ -lm

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
