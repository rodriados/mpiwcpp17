FROM ubuntu:20.04

MAINTAINER Rodrigo Siqueira <rodriados@gmail.com>

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y build-essential openmpi-bin openmpi-common libopenmpi3

COPY src /usr/local/include

ARG USER=mpiwcpp17
RUN adduser ${USER}

USER ${USER}
WORKDIR /home/${USER}
