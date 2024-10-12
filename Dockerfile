# Use the base image from the IHaskell notebook container registry
FROM ghcr.io/ihaskell/ihaskell-notebook:master

USER root

# RUN echo $(ls)

#ghc 9.4.5

# RUN mkdir /home/$NB_USER/hMPC
# COPY . /home/$NB_USER/src
# RUN rm /home/$NB_USER/stack.yaml

RUN echo -e "resolver: nightly-2023-05-17\npackages: []" > /home/$NB_USER/stack.yaml

# RUN stack ghci Runtime
# RUN echo $(which ghc)
# RUN echo "current" && pwd
# RUN echo $(stack ls ghc)
# RUN stack setup

# RUN stack install --global hMPC

# RUN stack install optparse-applicative

# RUN chown --recursive $NB_UID:users /home/$NB_USER/src
# RUN chown $NB_UID:users /home/$NB_USER/demos/Id3gini.ipynb

# COPY . /home/$NB_USER/hMPC


# WORKDIR /home/$NB_USER/hmpc
# RUN stack build hMPC
# WORKDIR /home/$NB_USER
# RUN stack install optparse-applicative \
#         hMPC

COPY demos /home/$NB_USER/demos
COPY app /home/$NB_USER/app
COPY test /home/$NB_USER/test
COPY stack.yaml /home/$NB_USER/stack.yaml
COPY package.yaml /home/$NB_USER/package.yaml

RUN chown --recursive $NB_UID:users /home/$NB_USER

USER $NB_UID

RUN stack build


ENV JUPYTER_ENABLE_LAB=yes
# RUN git clone https://github.com/nickvgils/hMPC.git /home/$NB_USER/