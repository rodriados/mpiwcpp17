# A thin C++17 wrapper for MPI.
# @file Makefile for building and automated testing
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2022-present Rodrigo Siqueira
NAME = mpiwcpp17

INCDIR = src
SRCDIR = src
TSTDIR = test

DSTDIR ?= dist
OBJDIR ?= obj
BINDIR ?= bin
PT3DIR ?= thirdparty

MPICXX ?= mpic++
STDCPP ?= c++17

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS     ?=
CXXFLAGS  ?= -std=$(STDCPP) -I$(DSTDIR) -I$(INCDIR) $(FLAGS)
LINKFLAGS ?= $(FLAGS)

SRCFILES := $(shell find $(SRCDIR) -name '*.h')                                \
            $(shell find $(SRCDIR) -name '*.hpp')

TSTFILES := $(shell find $(TSTDIR) -name '*.cpp')
TESTOBJS := $(TSTFILES:$(TSTDIR)/%.cpp=$(OBJDIR)/$(TSTDIR)/%.o)

# The operational system check. At least for now, we assume that we are always running
# on a Linux machine. Therefore, a disclaimer must be shown if this is not true.
SYSTEMOS := $(shell uname)
SYSTEMOS := $(patsubst MINGW%,Windows,$(SYSTEMOS))
SYSTEMOS := $(patsubst MSYS%,Msys,$(SYSTEMOS))
SYSTEMOS := $(patsubst CYGWIN%,Msys,$(SYSTEMOS))

ifneq ($(SYSTEMOS), Linux)
  $(info Warning: This makefile assumes OS to be Linux.)
endif

# If running an installation target, a prefix variable is used to determine where
# the files must be copied to. In this context, a default value must be provided.
ifeq ($(PREFIX),)
	PREFIX := /usr/local
endif

all:   tests
tests: build-tests

prepare-distribute:
	@mkdir -p $(DSTDIR)

export DISTRIBUTE_DESTINATION ?= $(shell realpath $(DSTDIR))

MPIWCPP17_DIST_CONFIG ?= .packconfig
MPIWCPP17_DIST_TARGET ?= $(DISTRIBUTE_DESTINATION)/$(NAME).h

distribute: prepare-distribute thirdparty-distribute $(MPIWCPP17_DIST_TARGET)
no-thirdparty-distribute: prepare-distribute $(MPIWCPP17_DIST_TARGET)

clean-distribute: thirdparty-clean
	@rm -rf $(DSTDIR)

export INSTALL_DESTINATION ?= $(PREFIX)/include
INSTALL_TARGETS = $(SRCFILES:$(SRCDIR)/%=$(INSTALL_DESTINATION)/%)

install: thirdparty-install $(INSTALL_TARGETS)

$(INSTALL_DESTINATION)/%: $(SRCDIR)/%
	install -m 644 -D -T $< $@

uninstall:
	@rm -f $(INSTALL_TARGETS)

prepare-tests:
	@mkdir -p $(BINDIR)/$(TSTDIR)
	@mkdir -p $(sort $(dir $(TESTOBJS)))

build-tests: override FLAGS := -DTESTING -g -O0 $(FLAGS)
build-tests: thirdparty-distribute prepare-tests $(BINDIR)/$(TSTDIR)/runtest

run-tests: NP ?= $(shell nproc)
run-tests: build-tests
	mpirun --host localhost:$(NP) -np $(NP) $(BINDIR)/$(TSTDIR)/runtest $(SCENARIO)

clean: clean-distribute
	@rm -rf $(BINDIR)
	@rm -rf $(OBJDIR)

.PHONY: all clean install uninstall
.PHONY: prepare-distribute distribute no-thirdparty-distribute clean-distribute
.PHONY: prepare-tests build-tests tests run-tests

$(MPIWCPP17_DIST_TARGET): $(SRCFILES)
	@python3 pack.py -c $(MPIWCPP17_DIST_CONFIG) -o $@

# Creates dependency on header files. This is valuable so that whenever a header
# file is changed, all objects depending on it will be recompiled.
ifneq ($(wildcard $(OBJDIR)/.),)
-include $(shell find $(OBJDIR) -name '*.d')
endif

$(BINDIR)/$(TSTDIR)/runtest: $(TESTOBJS)
	$(MPICXX) $(LINKFLAGS) $^ -o $@

$(OBJDIR)/$(TSTDIR)/%.o: $(TSTDIR)/%.cpp
	$(MPICXX) $(CXXFLAGS) -I$(TSTDIR) -MMD -c $< -o $@

# The target path for third party dependencies' distribution files. As each dependency
# may allow different settings, a variable for each one is needed.
THIRDPARTY_IGNORE ?=
THIRDPARTY_DEPENDENCIES = reflector

THIRDPARTY_TARGETS := $(filter-out $(THIRDPARTY_IGNORE),$(THIRDPARTY_DEPENDENCIES))
THIRDPARTY_TARGETS := $(THIRDPARTY_TARGETS:%=$(DISTRIBUTE_DESTINATION)/%.h)

thirdparty-distribute: prepare-distribute $(THIRDPARTY_TARGETS)
thirdparty-install:    $(THIRDPARTY_DEPENDENCIES:%=thirdparty-install-%)
thirdparty-uninstall:  $(THIRDPARTY_DEPENDENCIES:%=thirdparty-uninstall-%)
thirdparty-clean:      $(THIRDPARTY_DEPENDENCIES:%=thirdparty-clean-%)

ifndef MPIWCPP17_DIST_STANDALONE

thirdparty-distribute-%: $(DISTRIBUTE_DESTINATION)/%.h

$(DISTRIBUTE_DESTINATION)/%.h: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< distribute

thirdparty-install-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< install

thirdparty-uninstall-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< uninstall

thirdparty-clean-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< clean

else
.PHONY: $(THIRDPARTY_TARGETS)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-distribute-%)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-install-%)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-uninstall-%)
.PHONY: $(THIRDPARTY_DEPENDENCIES:%=thirdparty-clean-%)
endif

.PHONY: thirdparty-distribute thirdparty-install thirdparty-uninstall thirdparty-clean
.PHONY: $(THIRDPARTY_DEPENDENCIES)
