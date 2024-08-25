# Use the base image from the IHaskell notebook container registry
FROM ghcr.io/ihaskell/ihaskell-notebook:master

USER root

RUN rm -rf /home/$NB_USER/ihaskell_examples /home/$NB_USER/work

COPY . /home/$NB_USER

RUN chown $NB_UID:users /home/$NB_USER/demos/Id3gini.ipynb

USER $NB_UID

ENV JUPYTER_ENABLE_LAB=yes
# RUN git clone https://github.com/nickvgils/hMPC.git /home/$NB_USER/