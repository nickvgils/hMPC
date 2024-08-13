FROM ghcr.io/ihaskell/ihaskell-notebook:master

# Expose the default Jupyter Notebook port
EXPOSE 8888


# Start Jupyter Notebook (already set up in the base image)
CMD ["start-notebook.sh"]