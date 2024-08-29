---
title: Download .gguf Models From Ollama Library
date: 2024-07-26
---

GGUF is the model format required by tools based on [llama.cpp](https://github.com/ggerganov/llama.cpp) (such as
[koboldcpp](https://github.com/LostRuins/koboldcpp)). But how do we get models in this format?

One way to go is by downloading it directly from [Hugging Face](https://huggingface.co/models?library=gguf), but
unfortunately, there aren’t many .gguf models there; the majority are .safetensors. While we can [convert](https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py)
it, it's more convenient to get it directly in the desired format.

But Hugging Face is not the only repository for open models. We can rely on
[Ollama’s library](https://ollama.com/library) as well. By tinkering with its registry a bit, we can perform a direct
download of a .gguf file (without having Ollama installed).

## Step 1: Get a model

Go to the Ollama library page and pick the model you want to download. Note down the model name and parameters, as
you'll need them in the next steps:

![](/images/download-gguf-models-from-ollama-library/library.png)

## Step 2: Get the digest from the manifest

Navigate to the following URL, replacing `MODEL_NAME` and `PARAMETERS` with the values from Step 1:

```
https://registry.ollama.ai/v2/library/MODEL_NAME/manifests/MODEL_PARAMETERS
```

Example:

```
https://registry.ollama.ai/v2/library/phi3/manifests/3.8b
```

This URL will return a JSON object. Look for the digest field in the JSON and note down its value, as you'll need it in
the next step:

![](/images/download-gguf-models-from-ollama-library/digest.png)

## Step 3: Download the .gguf file

With the digest value from Step 2, replace `DIGEST` in the following URL and navigate to it:

```
https://registry.ollama.ai/v2/library/MODEL_NAME/blobs/sha256:DIGEST
```

Example:

```
https://registry.ollama.ai/v2/library/phi3/blobs/sha256:3e38718d00bb0007ab7c0cb4a038e7718c07b54f486a7810efd03bb4e894592a
```

This URL should initiate the download of a file named "data". This file is your .gguf model file, rename it as you
please and have fun!

