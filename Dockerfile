FROM continuumio/miniconda:latest

MAINTAINER ribin chalumattu <cribin@inf.ethz.ch>

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install mesa-utils -y

COPY ./weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5

# set work directory
WORKDIR /aestheticScorePredictor

# copy requirements.txt
COPY ./environment.yml /aestheticScorePredictor/environment.yml

RUN conda env create -f environment.yml

#RUN echo "source activate mlsp_environment" &gt; ~/.bashrc
ENV PATH /opt/conda/envs/mlsp_environment/bin:$PATH

# copy project
COPY . .

# set app port
EXPOSE 5001

#ENTRYPOINT [ "python3" ]
#ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "src.mlsp_predictor_server:app"]
ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "--env", "API_CONFIG=api_config.json", "aesthetics_predictor_api_pkg.predictor_api_server:app"]
