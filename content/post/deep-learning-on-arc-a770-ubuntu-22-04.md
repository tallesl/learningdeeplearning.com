---
title: Deep Learning on Arc A770 + Ubuntu 22.04
date: 2024-07-19
---

![](/images/deep-learning-on-arc-a770-ubuntu-22-04/arc-a770.png)

The graphics processor consumer market is no longer dominated by the two-brand dispute of NVIDIA's GeForce vs AMD's
Radeon. In 2022, a third contender entered the ring: Intel's Arc.

It is still a relatively unknown option to most desktop enthusiasts, which may explain why it is priced lower than its
competitors. For folks like me, trying to get as much VRAM as possible on a budget, is hard to dismiss.

But in a world where NVIDIA is the de facto standard and its seasoned AMD rival is struggling to make a dent, will the
support for Intel GPUs be there?

Are the drivers stable enough? Are the open-source project maintainers making the effort to support it? Let's find out.

## Downloading and Setting Up Drivers

A pleasant surprise here, there is no need to edit configuration files or install any package, there's out-of-the-box
support for it with the latest Linux kernel. The support is lackluster though: I couldn't get fan speed or temperature
sensors to work ([someone already asked for it](https://gitlab.freedesktop.org/drm/i915/kernel/-/issues/11276)).

## Checking GPU Usage

To make sure the GPU is working properly and to check its performance, we are going to install:

```
$ sudo apt install intel-gpu-tools

$ sudo apt install mesa-utils
```

Now let's fire glxgears while watching the GPU activity:

```
$ sudo intel_gpu_top

$ vblank_mode=0 glxgears -info
```

![](/images/deep-learning-on-arc-a770-ubuntu-22-04/glxgears.png)

The **“vblank_mode=0”** variable disables vsync, allowing glxgears to run free from syncing it with the display refresh rate.

We can see activity on the GPU, but I didn't make my card fans spin. Let's try with glmark2:

![](/images/deep-learning-on-arc-a770-ubuntu-22-04/glmark2.png)

That managed to get the GPU fan to spin.

## Ollama

After Ollama finished installing, it printed "AMD GPU ready" and then proceeded to use my Ryzen 5 5500 for
inference, yielding merely ~13 tokens per second (for comparison, I get ~100 tokens/s and ~75 tokens/s with my RTX 2060
and RX 7600 respectively).

Fingers crossed for the [merge request that I found adding support for Intel Arc](https://github.com/ollama/ollama/pull/2458)
to get merged soon.

Intel provides a custom Docker image (with custom scripts inside) for running Ollama, but I didn't have much
luck with it either.

Pulling `intelanalytics/ipex-llm-inference-cpp-xpu` image and running a container from it:

```
$ docker run -it --rm --net host --device /dev/dri --privileged --memory 32gb --shm-size 16gb intelanalytics/ipex-llm-inference-cpp-xpu
```

Argument            | Description
--------            | -----------
`-it`               | starts an interactive terminal session in the Docker container
`--rm`              | automatically removes the container and its filesystem after it exits
`--net host`        | uses the host network stack
`--device /dev/dri` | grants the container access to the Direct Rendering Infrastructure (DRI) devices on the host
`--privileged`      | gives extended privileges to the container, such as accessing all devices on the host system
`--memory 32gb`     | limits the memory usage of the container to 32 GB
`--shm-size 16gb`   | "shm" stands for shared memory, `/dev/shm` is 64 MB by default, which is little for applications with large datasets or heavy inter-process communication

Inside the container:

```
# . ipex-llm-init --gpu --device Arc
(...)
# sh /llm/scripts/start-ollama.sh
```

Then the error:

```
Error: llama runner process has terminated: signal: bus error (core dumped) 
```

Unfortunately, I gave up running Ollama on it for now.


## PyTorch

Following Intel's official documentation, we get this for installing the package:

```
$ pip install torch==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

And an error right after importing the package:

```
$ python3
Python 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import intel_extension_for_pytorch as ipex
Traceback (most recent call last):
(...)
OSError: libmkl_intel_lp64.so.2: cannot open shared object file: No such file or directory
```

Once again, let's resort to an Intel-provided Docker image:

```
$ docker run -it --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host intel/intel-extension-for-pytorch:2.1.30-xpu
```

Inside the container:

```
# python3
>>> import intel_extension_for_tensorflow as ipex
>>> print(ipex.__version__)
2.1.30+xpu
```

Looks OK.

## TensorFlow

Following Intel's official documentation, we get this for installing the package:

```
$ pip install intel-extension-for-tensorflow[xpu]
```

Then the following for checking the installation:

```py
import intel_extension_for_tensorflow as itex
print(itex.__version__)
```

And, of course, it didn't work:

```
tensorflow.python.framework.errors_impl.NotFoundError: libimf.so: cannot open shared object file: No such file or directory
```

Again, Docker to the rescue:

```
$ docker run -it --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host intel/intel-extension-for-tensorflow:xpu
```

While inside the container it seems that it's working as expected:

```py
>>> print(itex.__version__)
2.15.0.0
```

## Sources

- [Intel's Arc A770 16GB graphics card reaches $300 in the US, as DX11 performance rises by 19 percent](https://www.rockpapershotgun.com/intels-arc-a770-16gb-graphics-card-reaches-300-in-the-us-as-dx11-performance-rises-by-19-percent)
- [intel's Profile | Docker Hub](https://hub.docker.com/u/intel)
- [intelanalytics's Profile | Docker Hub](https://hub.docker.com/u/intelanalytics)
- [Introducing the Intel Extension for PyTorch for GPUs](https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-intel-extension-for-pytorch-for-gpus.html)
- [An Easy Introduction to Intel Extension for TensorFlow](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-intel-extension-for-tensorflow.html)
