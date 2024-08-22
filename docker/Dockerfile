FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y build-essential

RUN pip install nnunetv2

WORKDIR /app

COPY . /app

ENV nnUNet_raw="/app/nnunet_raw"
ENV nnUNet_preprocessed="/app/nnunet_preprocessed"
ENV nnUNet_results="/app/nnunet_results"