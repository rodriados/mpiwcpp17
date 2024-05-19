# A thin C++17 wrapper for MPI.
# @file Makefile for building and automated testing
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2022-present Rodrigo Siqueira
NAME = mpiwcpp17

INCDIR  = src
SRCDIR  = src
TESTDIR = test

DSTDIR ?= dist
OBJDIR ?= obj
BINDIR ?= bin
PT3DIR ?= thirdparty

GCPP   ?= mpic++
STDCPP ?= c++17

# Defining macros inside code at compile time. This can be used to enable or disable
# certain features on code or affect the projects compilation.
FLAGS ?=
GCPPFLAGS ?= -std=$(STDCPP) -I$(INCDIR) -I$(TESTDIR) $(FLAGS)
LINKFLAGS ?= $(FLAGS)

SRCFILES := $(shell find $(SRCDIR) -name '*.h')                                \
            $(shell find $(SRCDIR) -name '*.hpp')
TESTFILES := $(shell find $(TESTDIR) -name '*.cpp')
TESTDEPS = $(TESTFILES:$(TESTDIR)/%.cpp=$(OBJDIR)/%.test.o)

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

all: testing

prepare-distribute:
	@mkdir -p $(DSTDIR)

MPIWCPP17_DIST_CONFIG ?= .packconfig
MPIWCPP17_DIST_TARGET ?= $(DIST)/$(NAME).h

distribute: prepare-distribute thirdparty-distribute $(MPIWCPP17_DIST_TARGET)
no-thirdparty-distribute: prepare-distribute $(MPIWCPP17_DIST_TARGET)

clean-distribute: thirdparty-clean
	@rm -f $(MPIWCPP17_DIST_TARGET)
	@rm -rf $(DSTDIR)

INSTALL_DESTINATION ?= $(DESTDIR)$(PREFIX)/include
INSTALL_TARGETS = $(SRCFILES:$(SRCDIR)/%=$(INSTALL_DESTINATION)/%)

install: thirdparty-install $(INSTALL_TARGETS)

$(INSTALL_DESTINATION)/%: $(SRCDIR)/%
	install -m 644 -D -T $< $@

uninstall: thirdparty-uninstall
	@rm -f $(INSTALL_TARGETS)

prepare-build:
	@mkdir -p $(BINDIR)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(sort $(dir $(TESTDEPS)))

testing: override FLAGS = -g -O0
testing: prepare-build $(BINDIR)/runtest.o

runtest: testing
	mpirun --host localhost:$(np) -np $(np) $(BINDIR)/runtest.o $(scenario)

clean: clean-distribute
	@rm -rf $(BINDIR)
	@rm -rf $(OBJDIR)

.PHONY: all install prepare-build testing runtest uninstall clean
.PHONY: prepare-distribute distribute no-thirdparty-distribute

$(MPIWCPP17_DIST_TARGET): $(SRCFILES)
	python pack.py -c $(MPIWCPP17_DIST_CONFIG) -o $@

# Creates dependency on header files. This is valuable so that whenever a header
# file is changed, all objects depending on it will be recompiled.
ifneq ($(wildcard $(OBJDIR)/.),)
-include $(shell find $(OBJDIR) -name '*.d')
endif

$(BINDIR)/runtest.o: $(TESTDEPS)
	$(GCPP) $(LINKFLAGS) $^ -o $@

$(OBJDIR)/%.test.o: $(TESTDIR)/%.cpp
	$(GCPP) $(GCPPFLAGS) -MMD -c $< -o $@

# The target path for third party dependencies' distribution files. As each dependency
# may allow different settings, a variable for each one is needed.
export SUPERTUPLE_DIST_TARGET ?= $(DSTDIR)/supertuple.hpp
export REFLECTOR_DIST_TARGET  ?= $(DSTDIR)/reflector.hpp

THIRDPARTY_DEPENDENCIES ?= supertuple reflector
THIRDPARTY_TARGETS = $(SUPERTUPLE_DIST_TARGET) $(REFLECTOR_DIST_TARGET)

thirdparty-distribute: prepare-distribute $(THIRDPARTY_TARGETS)
thirdparty-install:    $(THIRDPARTY_DEPENDENCIES:%=thirdparty-install-%)
thirdparty-uninstall:  $(THIRDPARTY_DEPENDENCIES:%=thirdparty-uninstall-%)
thirdparty-clean:      $(THIRDPARTY_DEPENDENCIES:%=thirdparty-clean-%)

export REFLECTOR_DIST_STANDALONE = 1

$(SUPERTUPLE_DIST_TARGET):
ifndef SKIP_SUPERTUPLE_DISTRIBUTE
	@$(MAKE) --no-print-directory -C $(PT3DIR)/supertuple distribute
	cp $(PT3DIR)/supertuple/$@ $@
endif

$(REFLECTOR_DIST_TARGET):
ifndef SKIP_REFLECTOR_DISTRIBUTE
	@$(MAKE) --no-print-directory -C $(PT3DIR)/reflector distribute
	cp $(PT3DIR)/reflector/$@ $@
endif

thirdparty-install-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< install

thirdparty-uninstall-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< uninstall

thirdparty-clean-%: %
	@$(MAKE) --no-print-directory -C $(PT3DIR)/$< clean

.PHONY: thirdparty-distribute thirdparty-install thirdparty-uninstall thirdparty-clean
.PHONY: $(THIRDPARTY_DEPENDENCIES)
