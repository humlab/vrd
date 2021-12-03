FROM jupyter/tensorflow-notebook

USER root

RUN apt-get update \
 && apt-get -y dist-upgrade --no-install-recommends \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Imagemagick
RUN apt-get update && \
    apt-get install -y --no-install-recommends imagemagick ffmpeg libgavl1 frei0r-plugins && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mamba install --quiet --yes 'conda-forge::blas=*=openblas' 'matplotlib-base'   jupytext jupyterlab_widgets
RUN pip install ipyaggrid pandas "python-socketio[client]" aquirdturtle_collapsible_headings holoviews ipycytoscape ipysheet keract opencv-python Wand ffmpeg-python welford jupytext --upgrade 

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager 

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions /home/$NB_USER

RUN jupyter labextension install ipyaggrid
RUN jupyter lab build 

RUN npm cache clean --force && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    rm -rf /home/$NB_USER/.node-gyp && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
	
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions /home/$NB_USER

RUN mamba install --quiet --yes 'conda-forge::ffmpeg-python' pytorch faiss-cpu
