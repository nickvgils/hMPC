# Use the base image from the IHaskell notebook container registry
FROM ghcr.io/ihaskell/ihaskell-notebook:master

WORKDIR /home/$NB_USER

RUN rm -rf /home/$NB_USER/ihaskell_examples /home/$NB_USER/work

RUN git clone https://github.com/nickvgils/hMPC.git /home/$NB_USER/