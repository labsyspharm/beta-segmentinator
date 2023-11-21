FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get upgrade -y && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get upgrade -y
RUN apt-get install -y python3.10-full python3-pip git && git clone -b gmm https://github.com/labsyspharm/beta-segmentinator.git && pip install -r --user beta-segmentinator/requirements.txt

CMD ["python3.10", "beta-segmentinator/main.py"]