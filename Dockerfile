# Use the base image from the IHaskell notebook container registry
FROM ghcr.io/ihaskell/ihaskell-notebook:master

RUN rm -rf /home/$NB_USER/ihaskell_examples /home/$NB_USER/work

COPY . /home/$NB_USER

RUN chown --recursive $NB_UID:users /home/$NB_USER

# RUN git clone https://github.com/nickvgils/hMPC.git /home/$NB_USER/