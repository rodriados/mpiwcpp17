# A thin C++17 wrapper for MPI.
# @file Makefile for building and automated testing
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2022-present Rodrigo Siqueira
NAME = mpiwcpp17

INCDIR  = src
SRCDIR  = src
TGTDIR  = bin
TESTDIR = test

PWD = $(shell pwd)

GCPP   ?= mpic++
STDCPP ?= c++17

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=

GCPPFLAGS  = -std=$(STDCPP) -I$(INCDIR) $(FLAGS)
TESTFILES := $(shell find $(TESTDIR) -name '*.cpp')

all: testing

install: $(TGTDIR)

testing: override FLAGS = --coverage
testing: install $(TGTDIR)/runtest.o

runtest: testing coverage.info

clean:
	@rm -rf $(TGTDIR)
	@rm -f *.gcno *.gcda *.gcov
	@rm -f coverage.info

$(TGTDIR):
	@mkdir -p $@

$(TGTDIR)/runtest.o: $(TESTFILES)
	$(GCPP) $(GCPPFLAGS) $^ -o $@

coverage.info: $(TGTDIR)/runtest.o
	mpirun --host localhost:$(np) -np $(np) $< $(scenario)
	lcov -c -d . --no-external --exclude "$(PWD)/test/*" -o $@
	@sed -i 's|SF:$(PWD)/|SF:|g' $@

.PHONY: all install testing runtest clean
.PHONY: coverage.info
