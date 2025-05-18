#!/usr/bin/env bash
# A thin C++17 wrapper for MPI.
# @file The MPI emulator script for running tests with MPI locally.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
set -e

readonly USE_MPI_PROCESSES="${USE_MPI_PROCESSES:-4}"
readonly KNOWN_MPI_RUNNERS=(
  "mpiexec" "mpiexec.hydra" "mpiexec.mpd"
  "mpirun"
  "lamexec"
  "srun"
)

# Find what MPI executable is installed on the system and run the requested test.
# Although there are many name possibilities for the MPI runner, only `mpiexec`
# is guaranteed to behave as described in the standard.
for mpirunner in "${KNOWN_MPI_RUNNERS[@]}"; do
  if command -v "$mpirunner"; then
    exec $mpirunner -n $USE_MPI_PROCESSES "$@"
  fi
done

# If no MPI executable was found, then we must fail and inform this fact as the
# failure cause. This error should indicate that MPI is not correctly installed.
echo "MPI executable was not found."
exit 1
