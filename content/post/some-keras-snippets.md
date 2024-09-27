---
title: Some Keras Snippets
date: 2024-09-28
---

Here are some useful snippets that I've been using pretty much every time I play with Keras.

## Visualizing the model

You can get a model summary directly in the CLI, listing its layers, shapes, and parameters:

```
model.summary()
```

Output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ scale_to_0_1_range (Rescaling)  │ (None, 784)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ hidden_1 (Dense)                │ (None, 300)            │       235,500 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ hidden_2 (Dense)                │ (None, 100)            │        30,100 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output (Dense)                  │ (None, 10)             │         1,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 266,610 (1.02 MB)
 Trainable params: 266,610 (1.02 MB)
 Non-trainable params: 0 (0.00 B)
```

You can also get the summary as a PNG file:

```py
keras.plot_model(model, to_file='model.png')
```

![](/images/some-keras-snippets/model.png)

## Setting up a seed for reproducibility

Since we use a pseudo-random number generator (PRNG), setting up an initial seed value ensures reproducibility. This guarantees that the same random numbers are generated in different executions of the code.

Random numbers are used in many parts of the training process, such as the initialization of the weights, the shuffling of the data, and the dropout layers. Setting up the same seed ensures that the results are the same every time you run the same code with the same dataset.

You can set by calling:

```py
seed = 123

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
```

Or in a much handier way, just call:

```py
keras.utils.set_random_seed(123)
```

## Preventing TensorFlow from allocating all GPU memory

By default, TensorFlow allocates the entire available GPU memory at startup. To prevent this, call the following:

```py
gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
```

## Checking VRAM usage

To check the amount of VRAM used by TensorFlow on the spot, call:

```py
tf.config.experimental.get_memory_info('GPU:0')
```

Replace `GPU:0` with the desired GPU identifier.

## Setting up a custom callback

You can create a custom callback by subclassing the `Callback` class from `keras.callbacks`. Below are all the methods that can be overridden (override only the ones you need):

```py
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass
```

## Shut up TensorFlow

When importing TensorFlow, it often generates excessive log messages:

```
2024-09-15 12:58:03.328316: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-09-15 12:58:03.340620: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-09-15 12:58:03.344415: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-09-15 12:58:03.354197: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-09-15 12:58:03.955991: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1726415884.584769	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.621959	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.623432	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.625631	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.627330	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.629009	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.777146	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.778641	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1726415884.780130	6242 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-09-15 12:58:04.781466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3984 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2060, pci bus id: 0000:10:00.0, compute capability: 7.5
```

To suppress these logs, set the following environment variable (in your `.bashrc` file):

```
export TF_CPP_MIN_LOG_LEVEL=3
```

And for the "NUMA node" message, set up the following to run at system start:

```
echo 0 > /sys/bus/pci/devices/0000:10:00.0/numa_node
```

The `0000:10:00.0` identifier was taken by checking `lspci -D | grep NVIDIA` (check the one saying "VGA compatible controller").

## Sources

- [`summary` method](https://keras.io/api/models/model/#summary-method)
- [Model plotting utilities](https://keras.io/api/utils/model_plotting_utils/)
- [ tf.keras.utils.set\_random\_seed](https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed)
- [tf.config.experimental.set\_memory\_growth](https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth)
- [tf.config.experimental.get\_memory\_info](https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_memory_info)
- [Writing your own callbacks](https://keras.io/guides/writing_your_own_callbacks/)
- [TF_LogLevel](https://github.com/tensorflow/tensorflow/blob/043c4d515f352eff052b42dfc5a4bf5fe0dc00f6/tensorflow/c/logging.h#L27-L32)
- [How to set the NUMA node for an (NVIDIA) GPU persistently? - Ask Ubuntu](https://askubuntu.com/q/1379119/153234)
