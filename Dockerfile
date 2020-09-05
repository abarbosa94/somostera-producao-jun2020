FROM continuumio/miniconda:latest
RUN apt-get update && apt-get install -y build-essential && apt-get install -y manpages-dev
run mkdir /workspace
workdir /workspace
ADD enviroment.yaml .
ADD part3 .
ADD data .
# create conda environment
RUN conda env create -f enviroment.yaml
RUN echo "source activate tera-env" > ~/.bashrc
ENV PATH /opt/conda/envs/tera-env/bin:$PATH
