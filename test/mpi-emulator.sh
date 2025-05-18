#!/bin/env bash
# A thin C++17 wrapper for MPI.
# @file The MPI emulator selector to enable tests to run with MPI.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
set -e

readonly SOURCEDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

# Log the command line to be executed by the emulator into a log file. Setting this
# variable may help debugging the test suite or emulator logic itself.
if [[ -n "$USE_MPI_EMULATOR_LOGFILE" ]]; then
  echo "$@" >> "$USE_MPI_EMULATOR_LOGFILE"
fi

# If the purpose of the current execution is not to run a test, then we must simply
# forward the execution to whatever it was. This enables tests to be listed.
if [[ ! -n "$SHOULD_ENABLE_MPI_EMULATOR" ]]; then
  if [[ ! -n "$CTEST_INTERACTIVE_DEBUG_MODE" ]]; then
    exec "$@"
  fi
fi

# Forward execution to the selected MPI emulator. If the required environment variable
# is not found, then we bail out with an error message. We could assume the local
# emulator as the default one, but we cannot guarantee that was the intent.
case "$USE_MPI_EMULATOR" in
  "local"  ) exec "$SOURCEDIR/emulators/mpi-local-emulator.sh"  "$@"  ;;
  "docker" ) exec "$SOURCEDIR/emulators/mpi-docker-emulator.sh" "$@"  ;;
esac

# If no MPI emulator was defined, then we must fail and inform this fact as the failure
# cause. This error should indicate that no MPI emulator is selected to run tests.
echo "MPI emulator unknown or undefined."
exit 1
