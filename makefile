# A thin C++17 wrapper for MPI.
# @file Makefile for building and automated testing
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2022-present Rodrigo Siqueira
NAME = mpiwcpp17

INCDIR  = src
SRCDIR  = src
TGTDIR  = bin
TESTDIR = test

GCPP   ?= mpic++
STDCPP ?= c++17

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=

GCPPFLAGS  = -std=$(STDCPP) -I$(INCDIR) $(FLAGS)
TESTFILES := $(shell find $(TESTDIR) -name '*.cpp')

all: testing

install: $(TGTDIR)

testing: override FLAGS = -g -O0
testing: install $(TGTDIR)/runtest.o

runtest: testing
	mpirun --host localhost:$(np) -np $(np) $(TGTDIR)/runtest.o $(scenario)

clean:
	@rm -rf $(TGTDIR)
	@rm -f *.gcno *.gcda *.gcov
	@rm -f *.info

$(TGTDIR):
	@mkdir -p $@

$(TGTDIR)/runtest.o: $(TESTFILES)
	$(GCPP) $(GCPPFLAGS) $^ -o $@

.PHONY: all install testing runtest clean
