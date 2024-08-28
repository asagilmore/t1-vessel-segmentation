FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y build-essential git

RUN git clone --branch clDice_fast https://github.com/asagilmore/nnUNet.git /app/nnUNet

RUN pip install -e /app/nnUNet

WORKDIR /app

COPY . /app

ENV nnUNet_raw="/app/nnunet_data/nnunet_raw"
ENV nnUNet_preprocessed="/app/nnunet_data/nnunet_preprocessed"
ENV nnUNet_results="/app/nnunet_data/nnunet_results"