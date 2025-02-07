#!/bin/bash
# A thin C++17 wrapper for MPI.
# @file The MPI emulator script for running tests with MPI.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
readonly quoted_arguments=$(printf '%q ' "$@")

# If the purpose of the current execution is to list the tests available, then we
# must simply pass the arguments over, run them and bail out.
for arg in $quoted_arguments; do
  if [ "$arg" = "--list-tests" ]; then
    eval "$quoted_arguments"
    exit 0
  fi
done

# Finds the MPI executable if installed on the system and runs the request test.
# Although there are many name possibilities for the MPI runner, only `mpiexec`
# is guaranteed to behave as described in the standard.
readonly known_mpi_runners=(
  "mpiexec" "mpiexec.hydra" "mpiexec.mpd"
  "mpirun"
  "lamexec"
  "srun"
)

for mpi_runner in "${known_mpi_runners[@]}"; do
  if command -v "$mpi_runner"; then
    eval "$mpi_runner -n $(nproc) --oversubscribe $quoted_arguments"
    exit $?
  fi
done

# If no MPI executable was found, then we must fail and inform this fact as the
# failure cause. This error should indicate that MPI is not correctly installed.
echo "MPI executable was not found."
exit 1
