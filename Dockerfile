# Use the base image from the IHaskell notebook container registry
FROM ghcr.io/ihaskell/ihaskell-notebook:master

USER root

# RUN echo $(ls)

#ghc 9.4.5

RUN echo -e "resolver: nightly-2023-05-17\npackages: []" > /home/$NB_USER/stack.yaml

COPY demos /home/$NB_USER/demos
COPY app /home/$NB_USER/app
COPY test /home/$NB_USER/test
COPY stack.yaml /home/$NB_USER/stack.yaml
COPY package.yaml /home/$NB_USER/package.yaml

RUN chown --recursive $NB_UID:users /home/$NB_USER

USER $NB_UID

RUN stack build --fast


ENV JUPYTER_ENABLE_LAB=yes