#!/bin/env bash
# A thin C++17 wrapper for MPI.
# @file The MPI emulator script for running tests with MPI in a Docker cluster.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2025-present Rodrigo Siqueira
set -e

readonly USE_MPI_PROCESSES="${USE_MPI_PROCESSES:-4}"
readonly USE_MPI_NETWORK="${USE_MPI_NETWORK:-mpinet}"
readonly USE_MPI_IMAGE="${USE_MPI_IMAGE:-mpinode}"
readonly USE_MPI_USER="${USE_MPI_USER:-mpiuser}"

# Find the IPs of all hosts connected to the MPI cluster network. Every host found
# on the cluster network will be considered as available for running tests.
# @return The list of hosts available for running MPI tests.
find_mpi_hosts() {
  local format="{{.NetworkSettings.Networks.$USE_MPI_NETWORK.IPAddress}}"
  for host in $(docker ps -q --filter network=$USE_MPI_NETWORK); do
    docker inspect --format "$format:$USE_MPI_PROCESSES" $host
  done
}

readonly MPI_HOST_LIST=$(find_mpi_hosts | paste -sd ',')

# Forward execution to the corresponding test through the MPI network cluster. In
# order to avoid possible conflicts with the MPI version installed in the host machine,
# the tests are invoked from an ephemeral container that communicate with all hosts
# previously found in the network.
exec docker run --rm --network $USE_MPI_NETWORK --user $USE_MPI_USER  \
  $USE_MPI_IMAGE mpirun --host "$MPI_HOST_LIST" -- "$@"
