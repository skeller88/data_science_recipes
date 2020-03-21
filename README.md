# Overview
There are a lot of data science and machine learning tutorials out there, with custom code for every problem. I haven't
yet found a repo that generalizes the code in these tutorials to make it easy to apply to new problems. As I tackle
more and more data science and machine learning problems, I'll continue adding to this repo. Contributions welcome.

Some of the code is meant to be copy and pasted into a jupyter notebook, like `imports.py`.

# Run a notebook with a conda environment via Docker
```
cp ./.bash/env_secrets.sample ./.bash/env_secrets
# Prepare a hashed password:
# https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#preparing-a-hashed-password
# set JUPYTER_PASSWORD_SHA to your hashed password
source .bash/env_secrets

export FILEDIR=conda_jupyter_notebook
export IMAGE_NAME=$BASE_IMAGE_NAME/conda_jupyter_notebook

docker build -t $IMAGE_NAME \
--file $FILEDIR/Dockerfile \
--build-arg jupyter_password_sha_build_arg=$JUPYTER_PASSWORD_SHA .

docker run -it --rm -p 8888:8888 \
--volume ~:/home/jovyan/work \
--env-file $FILEDIR/env.list $IMAGE_NAME

docker push $IMAGE_NAME
```
