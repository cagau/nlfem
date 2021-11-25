# Comment
FROM ubuntu
ARG PYTHON='/usr/bin/python3'
USER root
RUN apt-get update
RUN apt-get -yq install git gcc g++ libarmadillo-dev liblapack-dev libmetis-dev
RUN apt-get -yq install python3-pip python3-dev libgmp-dev libcgal-dev
RUN ${PYTHON} -m pip install jupyterlab
RUN jupyter notebook --generate-config
RUN mkdir nlfem
ADD . nlfem
RUN cp nlfem/docker/jupyter_notebook_config.py root/.jupyter/
WORKDIR nlfem
RUN ${PYTHON} -m pip install -r requirements.txt
RUN ${PYTHON} -m pip install RISE
RUN ${PYTHON} setup.py build --force install

