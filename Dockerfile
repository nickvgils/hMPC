# Use the base image from the IHaskell notebook container registry
FROM ghcr.io/ihaskell/ihaskell-notebook:master

RUN rm -rf /home/$NB_USER/ihaskell_examples /home/$NB_USER/work