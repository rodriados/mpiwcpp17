FROM ubuntu:latest

MAINTAINER Rodrigo Siqueira <rodriados@gmail.com>

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y build-essential git openmpi-bin openmpi-common

COPY src /usr/local/include

ARG USER=mpiwcpp17
RUN adduser ${USER}

USER ${USER}
WORKDIR /home/${USER}
