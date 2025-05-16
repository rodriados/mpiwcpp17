# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

RUN apt-get update
RUN apt-get install -y openssh-server
RUN apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

ARG USER=mpiuser
RUN useradd -ms /bin/bash ${USER}

RUN mkdir -p /home/${USER}/.ssh
RUN ssh-keygen -t rsa -N "" -f /home/${USER}/.ssh/id_rsa
RUN cat /home/${USER}/.ssh/id_rsa.pub >> /home/${USER}/.ssh/authorized_keys
RUN chmod 600 /home/${USER}/.ssh/authorized_keys
RUN chown -R ${USER}:${USER} /home/${USER}/.ssh

COPY sshconfig /home/${USER}/.ssh/config
RUN chown ${USER}:${USER} /home/${USER}/.ssh/config

RUN mkdir /var/run/sshd

CMD ["/usr/sbin/sshd", "-D"]
