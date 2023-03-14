FROM python:3.9-slim

#apt package install
RUN apt update -y && apt install curl unzip -y

#apt cache clean
RUN apt cache clean all

RUN mkdir /apprunner-inference

WORKDIR /apprunner-inference

COPY ./dataset ./dataset
COPY ./server.py .
COPY ./requirements.txt .

#dataset download
RUN cd ./dataset && chmod +x ./dataset_download.sh
RUN cd ./dataset && ./dataset_download.sh

#pip package install
RUN pip install -r requirements.txt

#pip cache clean
RUN rm -rf ~/.cache/pip/*

#first run (for model download)
RUN (nohup python server.py &) ; while ! curl -s --head --request GET "http://localhost:5001/healthcheck" | grep "200 OK" > /dev/null; do sleep 1; done;

EXPOSE 5001

CMD ["python", "server.py"]





