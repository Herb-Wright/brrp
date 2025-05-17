# Use NVIDIA's base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables to non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && \
 rm -rf /var/lib/apt/lists/* && \
 apt-get update -y && \
 apt-get install -y apt-transport-https ca-certificates && \
 apt-get update --allow-insecure-repositories && \
 apt-get install -y --allow-unauthenticated gnupg && \
 apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 871920D1991BC93C

# Install necessary packages including curl and software-properties-common
RUN apt-get update && apt-get install -y \
 lsb-release \
 curl \
 gnupg \
 sudo \
 ca-certificates \
 build-essential \
 cmake \
 git \
 wget \
 libopencv-dev \
 software-properties-common \
 locales \
 && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10 and related tools
RUN apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set Python 3.10 as the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10

# Ensure pip is properly linked
RUN python3.10 -m pip install --upgrade pip && \
    ln -sf $(which pip3) /usr/local/bin/pip

# Verify Python version
RUN python --version
# Minimal setup
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales
 
# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-noetic-desktop-full
RUN apt-get install -y --no-install-recommends python3-rosdep
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# # Install Conda (Miniconda recommended for smaller size)
# RUN curl -sS https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
#     && bash miniconda.sh -b -p /opt/conda \
#     && rm miniconda.sh

# # Set PATH for conda
# ENV PATH=/opt/conda/bin:$PATH



# # Create and activate the Conda environment
# RUN conda clean -a && \
#     for i in $(seq 1 3); do \
#       conda env create -f /brrp/environment.yml && break || \
#       echo "Retry attempt $i/3" && \
#       conda clean -a && \
#       sleep 5; \
#     done

# pip install dependencies
RUN pip install numpy
RUN pip install --ignore-installed open3d
RUN pip install --no-cache-dir torch==2.6
# RUN python -c "import torch; print(torch.__version__)"
RUN pip install --no-cache-dir torchvision
RUN pip install --ignore-installed torch_geometric
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+${CUDA}.html
RUN pip install matplotlib
RUN pip install trimesh
RUN pip install scikit-image
RUN pip install point-cloud-utils
RUN pip install transformers
# RUN pip install pyglet<2

# # Set the environment variable for the Conda environment
# ENV CONDA_DEFAULT_ENV=brrp
# ENV PATH=/opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# RUN /opt/conda/bin/conda run -n brrp pip install -e .

# # Install ROS dependencies for Catkin workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; \
    apt-get update; \
    apt-get install -y python3-catkin-tools"

# Copy your workspace into the container
RUN mkdir /catkin_ws
RUN mkdir /catkin_ws/src
COPY ./custom_msgs /catkin_ws/src

# # RUN apt-get install -y python3-empy
RUN pip install --no-cache-dir --ignore-installed empy==3.3.4 catkin_pkg rosnumpy gdown filelock

# # get the right torch
# RUN /opt/conda/bin/conda run -n brrp conda remove -y pytorch && \
#     /opt/conda/bin/conda run -n brrp pip install --no-cache-dir torch --no-deps --ignore-installed

# # Create a working directory for your project
WORKDIR /brrp

# # Copy your environment.yml file for Conda environment setup
COPY . .

# TODO: should I remove -e?
RUN pip install .

# # Initialize the Catkin catkin_ws and build custom_msgs
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; \
    cd /catkin_ws && \
    catkin_make"

# Expose necessary ROS ports (you may need additional ports depending on your project)
EXPOSE 11311

RUN echo "source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash" >> ~/.bashrc

RUN python -m gdown https://drive.google.com/uc?id=1h9kdW4QKFN3EzeOarZ82SOiiXu-T6ssw -O ycb_prior.tar.xz
RUN tar -xf ycb_prior.tar.xz

# # Default command (you can change this based on how you run your ROS node)
CMD ["/bin/bash", "-c"]