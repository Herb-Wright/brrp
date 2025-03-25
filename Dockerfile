# Use NVIDIA's base image for GPU support
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables to non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

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
    python3-pip \
    python3-dev \
    python3-venv \
    libopencv-dev \
    software-properties-common \
    locales \
    && rm -rf /var/lib/apt/lists/*

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

# Install Conda (Miniconda recommended for smaller size)
RUN curl -sS https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Set PATH for conda
ENV PATH=/opt/conda/bin:$PATH

# Create a working directory for your project
WORKDIR /brrp

# Copy your environment.yml file for Conda environment setup
COPY . .

# Create and activate the Conda environment
RUN conda env create -f /brrp/environment.yml \
    && conda clean -a

# Set the environment variable for the Conda environment
ENV CONDA_DEFAULT_ENV=brrp
ENV PATH=/opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

RUN /opt/conda/bin/conda run -n brrp pip install -e .

# Install ROS dependencies for Catkin workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; \
    apt-get update; \
    apt-get install -y python3-catkin-tools"

# Copy your workspace into the container
RUN mkdir /catkin_ws
RUN mkdir /catkin_ws/src
COPY ./custom_msgs /catkin_ws/src

# RUN apt-get install -y python3-empy
RUN /opt/conda/bin/conda run -n brrp pip install empy==3.3.4 catkin_pkg rosnumpy gdown

# get the right torch
RUN /opt/conda/bin/conda run -n brrp conda remove -y pytorch && \
    /opt/conda/bin/conda run -n brrp pip install torch


# Initialize the Catkin catkin_ws and build custom_msgs
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; \
    cd /catkin_ws && \
    catkin_make"

# # Expose necessary ROS ports (you may need additional ports depending on your project)
# EXPOSE 11311

RUN echo "source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash" >> ~/.bashrc

RUN /opt/conda/bin/conda run -n brrp python -m gdown https://drive.google.com/uc?id=1h9kdW4QKFN3EzeOarZ82SOiiXu-T6ssw -O ycb_prior.tar.xz
RUN tar -xf ycb_prior.tar.xz

# Default command (you can change this based on how you run your ROS node)
CMD ["/bin/bash", "-c"]