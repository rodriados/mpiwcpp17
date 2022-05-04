# MPI-wrapper for C++17
![license MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![version beta](https://img.shields.io/badge/version-beta-green.svg)

A thin and modern C++17 wrapper around the pure-C implementation for the
[Message Passing Interface](https://www.mpi-forum.org/) (MPI) standard.

## Motivation
Due to the lack of an official C++ specification or implementation for the MPI standard,
developers have no alternatives but to keep using the low-level C API to achieve
software parallelization with MPI. Although realible and extremely portable, the
functions defined by the standard are not really easy to use or understand.

This project focuses on wrapping the MPI standard functions into clean, elegant,
type-safe and user-friendly interfaces, by exploring modern C++17's language features
and idioms without the introduction of significant run-time overhead to the routines.
Nonetheless, this library has no aim to provide direct mappings of the C API into
C++ functions, namespaces or classes.

This library is best used by developers who are already familiar with the MPI standard
and functions, but is looking for a C++ idiomatic alternative. Be advised that not
every feature are guaranteed to be supported.

## Install
The library requires any MPI implementation and MPI-compatible C++-compiler to be
installed in your system. As a header-only library, you can directly download or
copy the files into your own project or clone it following the steps below:
```bash
git clone https://github.com/rodriados/mpiwcpp17
```

## Usage
To use the project, you can copy source files into your own project or install it
in your system and then reference it in your code:
```cpp
#include <mpiwcpp17.h>
```
