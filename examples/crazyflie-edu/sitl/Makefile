# Makefile for Crazyflie SITL
#
# Usage:
#   make          # Build SITL binary
#   make clean    # Clean build artifacts
#   make run      # Build and run

CC ?= clang
CFLAGS = -Wall -Wextra -O2 -g
LDFLAGS = -lm

# Source files
SRCS = sitl_main.c ../user_code.c
OBJS = $(SRCS:.c=.o)
TARGET = sitl_main

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)

