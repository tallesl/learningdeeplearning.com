---
title: Deep Learning on Arc A770 + Ubuntu 22.04 (work in progress)
date: 2024-07-17
---

![](/images/deep-learning-on-arc-a770-ubuntu-22-04/arc-a770.png)

The graphics processor consumer market is no longer dominated by the two-brand dispute of NVIDIA's GeForce vs AMD's Radeon. In 2022, a third contender entered the ring: Intel's Arc.

It is still a relatively unknown option to most desktop enthusiasts, which may explain why it is priced lower than its competitors as of now. For folks like me who are trying to get VRAM as much as possible on a budget, it's simply hard to dismiss it.

In a world in which NVIDIA is the de facto standard and its seasoned AMD rival is struggling to make a dent, is the support for Intel GPU there?

Are the drivers stable enough? Are the open-source project maintainers making the effort to support it? Let's find out in the sections below.

## Downloading and Setting Up Drivers

Works out of the box!

## Checking GPU Usage

```
$ sudo apt install intel-gpu-tools

$ sudo intel_gpu_top
```

## Ollama

After Ollama finished installing it printed "AMD GPU ready" and proceed to use the CPU.

https://github.com/ollama/ollama/pull/2458

## PyTorch

TBD

## TensorFlow

TBD

## Sources

- Intel's Arc A770 16GB graphics card reaches $300 in the US, as DX11 performance rises by 19 percent
