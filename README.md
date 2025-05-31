# Bayesian Reconstruction with Retrieval-Augmented Priors

![](./website/fig1.png)

Our method, BRRP, **(a)**, **(b)** takes an input segmented RGBD image, and **(c)** *retrieves* objects to use as a prior, which allow it to **(d)** reconstruct the scene as well as **(e)** capture principled *uncertainty* about object shape.

Here are some useful links for more information:

- [Website](https://herb-wright.github.io/brrp/)
- [Arxiv](https://arxiv.org/abs/2411.19461)

## Getting Started

### From Source

**Clone the repo:** start by making sure you have the repo cloned:

```sh
git clone https://github.com/Herb-Wright/brrp.git
cd brrp
```

**Dependencies:** if you want to be able to use the code, you need to install the dependencies, the best way to do this is to use a conda environment. If you have a cuda-enabled GPU:

```sh
conda env create -f environment.yml
conda activate brrp
```

Of course, you can also use `mamba` instead. If you *don't* have a cuda-enabled GPU or want to run the code on the CPU, you can use the `env_cpu.yml` file instead (creates a `brrp_cpu` environment).

**Installing Project:**

```sh
pip install .
```

**Running an Example:** To run the example script, make sure you have downloaded the scenes and have them in the `~/data/brrp_real_world_scenes` directory (or edit the directory in the script). You also need to have the ycb_prior in that same directory. Then, you can run the command:

```sh
python scripts/example.py
```

This should cause a pop-up visualization of the reconstructed meshes and point cloud (after BRRP method is run).

**Note:** for the other scripts, you may need to install additional dependencies. These *should be* all listed at the top of the script. You also may need to download datasets for them to run as well.

### Pip install from Github

Alternatively, if you have an environment, you can simply pip install directly from Github:

```sh
pip install https://github.com/Herb-Wright/brrp.git
```

Then, you can use it in python:

```py
from brrp.full_method import full_brrp_method

...

weights, hp_trans = full_brrp_method(
    rgb=rgb, 
    xyz=xyz, 
    mask=seg_map, 
    prior_path=abspath("~/data/ycb_prior"), 
    device_str=device_str
)
```

### Docker/Ros Version

Clone and build:

```sh
git clone https://github.com/Herb-Wright/brrp.git
cd brrp
docker build -t brrp .
```

Run the container:

```sh
docker run -it   --net=host   --gpus all   --env="NVIDIA_DRIVER_CAPABILITIES=all"   --env="DISPLAY=$DISPLAY"   --env="QT_X11_NO_MITSHM=1"   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"   --volume="$XAUTHORITY:/root/.Xauthority:rw"   brrp bash
```

In the container (This is for Martin only):

```sh
echo "155.98.68.120 perception-pc" >> /etc/hosts
export ROS_MASTER_URI=http://perception-pc:11311
export ROS_IP=haku.cs.utah.edu
```

Then to run the node:
```sh
python scripts/run_ros.py
# OR for PointSDF:
python scripts/run_ros_pointsdf.py
```

## Datasets

- YCB Prior - *COMING SOON*
- Shifted Scenes - *COMING SOON*
- PointSDF Weights - *COMING SOON*


## Citation

Please consider citing our work:

```
@article{wright2024robust,
  title={Robust Bayesian Scene Reconstruction by Leveraging Retrieval-Augmented Priors},
  author={Wright, Herbert and Zhi, Weiming and Johnson-Roberson, Matthew and Hermans, Tucker},
  journal={arXiv preprint arXiv:2411.19461},
  year={2024}
}
```


