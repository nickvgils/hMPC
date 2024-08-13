# Use the base image from the IHaskell notebook container registry
FROM ghcr.io/ihaskell/ihaskell-notebook:latest

# Install any additional dependencies needed for MyBinder
# Install git and nano for ease of use
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy any local files or Jupyter notebooks into the container
COPY . /home/jovyan/work/

# Set the working directory
WORKDIR /home/jovyan/work

# Install Python packages required by your notebooks
# Uncomment and edit the line below to install Python dependencies
# RUN pip install -r requirements.txt

# Set permissions to allow MyBinder to write to the work directory
RUN chown -R jovyan:users /home/jovyan/work

# Expose the default Jupyter port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''"]