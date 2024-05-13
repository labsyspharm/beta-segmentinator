FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get upgrade -y && apt-get install -y software-properties-common build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl wget \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && apt-get update && apt-get upgrade -y

RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.14

RUN apt update && apt-get install -y git  && git clone https://github.com/labsyspharm/beta-segmentinator.git && python3.10 -m pip install -r beta-segmentinator/requirements.txt

COPY model.pt /model.pt

CMD ["python3.10", "/beta-segmentinator/main.py"]
