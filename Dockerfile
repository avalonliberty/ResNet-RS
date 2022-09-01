FROM ubuntu:22.04
WORKDIR /RESNETRS
COPY . .
RUN apt-get update -y && \
    apt-get install -y python3.10 && \
    apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
