FROM continuumio/miniconda3:4.7.12

# https://askubuntu.com/questions/141928/what-is-the-difference-between-bin-sh-and-bin-bash
# bash has more functionality than bin, such as "source"
SHELL [ "/bin/bash", "--login", "-c" ]

# Must match conda location in base Dockerfile:
# https://github.com/ContinuumIO/docker-images/blob/master/miniconda/debian/Dockerfile
ENV CONDA_DIR /opt/conda

# make non-activate conda commands available
ENV PATH $CONDA_DIR/bin:$PATH

# Conda env installation and setup based on code from:
# https://github.com/kaust-vislab/tensorflow-gpu-data-science-project/blob/0ef82814ec1cc00c1817d2fed4328fcec885f647/docker/Dockerfile
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_NAME env

RUN cat ~/.bashrc
RUN echo $PATH

RUN conda create -n $ENV_NAME python=3.6 && \
    # https://automl.github.io/auto-sklearn/master/installation.html#installing-auto-sklearn
    conda install -n $ENV_NAME gxx_linux-64 gcc_linux-64 swig

COPY ./conda_jupyter_notebook/environment.yml ./environment.yml

# make conda activate command for $ENV_PREFIX environment available from /bin/bash --interactive shells
RUN echo "source activate $ENV_NAME" > ~/.bashrc

ENV PATH $CONDA_DIR/envs/$ENV_NAME/bin:$PATH

RUN conda update --name base --channel defaults conda && \
    conda env update -n $ENV_NAME --file ./environment.yml && \
    conda activate $ENV_NAME && \
    conda clean --all --yes

# RUN ls $CONDA_DIR/bin
RUN touch ~/.bashrc
RUN echo "source $CONDA_DIR/bin/activate $ENV_NAME" > ~/.bashrc
RUN conda init bash
# https://stackoverflow.com/questions/54429210/how-do-i-prevent-conda-from-activating-the-base-environment-by-default
RUN conda config --set auto_activate_base false
RUN cat ~/.bashrc

COPY . .

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Set default JUPYTER_PASSWORD_SHA at build time https://vsupalov.com/docker-arg-vs-env/
ARG jupyter_password_sha_build_arg
# Password to login to the jupyter notebook
ENV JUPYTER_PASSWORD_SHA $jupyter_password_sha_build_arg

CMD ["bash", "-c", "source ~/.bashrc && conda activate $ENV_NAME && jupyter notebook --ip 0.0.0.0 --allow-root \
--notebook-dir /home/jovyan/work --NotebookApp.password=$JUPYTER_PASSWORD_SHA"]
