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
