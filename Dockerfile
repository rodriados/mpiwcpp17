FROM alpine:latest

MAINTAINER Rodrigo Siqueira <rodriados@gmail.com>

ARG USER=mpiwcpp17

ARG OPENMPI_VERSION="4.1.3"
ARG OPENMPI_MAJOR_VERSION="v4.1"
ARG OPENMPI_CONFIGURE_OPTIONS
ARG OPENMPI_MAKE_OPTIONS="-j8"
ARG OPENMPI_MD5SUM="e0b9977385cec8d8e6320c4b75c65f6a"

RUN apk update && apk upgrade
RUN apk add --no-cache build-base libatomic perl linux-headers openssh

RUN mkdir -p /tmp/openmpi/src

WORKDIR /tmp/openmpi/src
RUN wget https://download.open-mpi.org/release/open-mpi/${OPENMPI_MAJOR_VERSION}/openmpi-${OPENMPI_VERSION}.tar.gz
RUN echo "${OPENMPI_MD5SUM}  openmpi-${OPENMPI_VERSION}.tar.gz" > openmpi.md5sum && md5sum -c openmpi.md5sum
RUN tar xfz openmpi-${OPENMPI_VERSION}.tar.gz

WORKDIR /tmp/openmpi/src/openmpi-${OPENMPI_VERSION}
RUN ./configure ${OPENMPI_CONFIGURE_OPTIONS}
RUN make all ${OPENMPI_MAKE_OPTIONS}
RUN make install

WORKDIR /
RUN rm -rf /tmp/openmpi
RUN addgroup -S ${USER} && adduser -S ${USER} -G ${USER}

COPY src /usr/local/include

WORKDIR /home/${USER}
USER ${USER}
