FROM ubuntu:20.04

MAINTAINER Rodrigo Siqueira <rodriados@gmail.com>

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y build-essential wget git lcov openmpi-bin openmpi-common libopenmpi3

WORKDIR /usr/local/include
RUN wget https://github.com/catchorg/Catch2/releases/download/v2.13.9/catch.hpp

ARG USER=mpiwcpp17
RUN adduser ${USER}

USER ${USER}
WORKDIR /home/${USER}
