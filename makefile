# A thin C++17 wrapper for MPI.
# @file Makefile for building and automated testing
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2022-present Rodrigo Siqueira
NAME = mpiwcpp17

INCDIR  = src
SRCDIR  = src
OBJDIR  = obj
TGTDIR  = bin
TESTDIR = test

GCPP   ?= mpic++
STDCPP ?= c++17

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=
GCPPFLAGS ?= -std=$(STDCPP) -I$(INCDIR) -I$(TESTDIR) $(FLAGS)
LINKFLAGS ?= $(FLAGS)

TESTFILES := $(shell find $(TESTDIR) -name '*.cpp')
TESTDEPS = $(TESTFILES:$(TESTDIR)/%.cpp=$(OBJDIR)/%.test.o)

all: testing

install:
	@mkdir -p $(TGTDIR)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(sort $(dir $(TESTDEPS)))

testing: override FLAGS = -g -O0
testing: install $(TGTDIR)/runtest.o

runtest: testing
	mpirun --host localhost:$(np) -np $(np) $(TGTDIR)/runtest.o $(scenario)

clean:
	@rm -rf $(TGTDIR)
	@rm -rf $(OBJDIR)

# Creates dependency on header files. This is valuable so that whenever a header
# file is changed, all objects depending on it will be recompiled.
ifneq ($(wildcard $(OBJDIR)/.),)
-include $(shell find $(OBJDIR) -name '*.d')
endif

$(TGTDIR)/runtest.o: $(TESTDEPS)
	$(GCPP) $(LINKFLAGS) $^ -o $@

$(OBJDIR)/%.test.o: $(TESTDIR)/%.cpp
	$(GCPP) $(GCPPFLAGS) -MMD -c $< -o $@

.PHONY: all install testing runtest clean
