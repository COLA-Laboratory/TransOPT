FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y git wget unzip build-essential

ENV INSTALL_DIR=/usr/local/Absolut
ENV TEMP_DIR=/root/Absolut_temp
ENV REPO_URL=https://github.com/csi-greifflab/Absolut

RUN mkdir -p $INSTALL_DIR && mkdir -p $TEMP_DIR

RUN git clone $REPO_URL $TEMP_DIR && \
    cd $TEMP_DIR/src && \
    sed -i 's/-Wl//g' Makefile && \
    make && \
    mv AbsolutNoLib /usr/local/bin/AbsolutNoLib

RUN rm -rf $TEMP_DIR

COPY prepare_antigen.sh /usr/local/bin/prepare_antigen.sh
RUN chmod +x /usr/local/bin/prepare_antigen.sh

WORKDIR $INSTALL_DIR