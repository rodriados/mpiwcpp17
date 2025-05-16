#!/bin/env bash
# A thin C++17 wrapper for MPI.
# @file The MPI emulator script for running tests with MPI.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
readonly QUOTED_ARGUMENTS=$(printf '%q ' "$@")
readonly KNOWN_MPI_RUNNERS=(
  "mpiexec" "mpiexec.hydra" "mpiexec.mpd"
  "mpirun"
  "lamexec"
  "srun"
)

# If the purpose of the current execution is to list the tests available, then we
# must simply pass the arguments over, run them and bail out.
for arg in $QUOTED_ARGUMENTS; do
  if [ "$arg" = "--list-tests" ]; then
    eval "$QUOTED_ARGUMENTS"
    exit 0
  fi
done

# Finds the MPI executable if installed on the system and runs the requested test.
# Although there are many name possibilities for the MPI runner, only `mpiexec`
# is guaranteed to behave as described in the standard.
for mpirunner in "${KNOWN_MPI_RUNNERS[@]}"; do
  if command -v "$mpirunner"; then
    eval "$mpirunner -n 4 --host localhost:4 $QUOTED_ARGUMENTS"
    exit $?
  fi
done

# If no MPI executable was found, then we must fail and inform this fact as the
# failure cause. This error should indicate that MPI is not correctly installed.
echo "MPI executable was not found."
exit 1
