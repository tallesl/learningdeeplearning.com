---
title: LLM Inference as a Service
date: 2024-07-29
---

Just like other "as-a-service" offerings, LLMs are following the trend, with multiple providers stepping up to simplify
their use and integration. Running large language models on your own can be complex—they typically require GPUs and
constantly evolving libraries and models. Leveraging a provider that offers out-of-the-box inference can remove much of
the incidental complexity, allowing you to focus on what’s essential for your use case.

## Pricing comparison (July 2024)

Here's a bird's-eye view of some top LLM service providers and their pricing:

Provider    | Llama 3 8B (per 1M tokens) | Mixtral-8X7B (per 1M tokens) | Pricing page
--------    | -------------------------- | ---------------------------- | ------------
Anyscale    | $0.15                      | $0.15                        | [anyscale.com](https://www.anyscale.com/pricing)
Deep Infra  | $0.06                      | $0.24                        | [deepinfra.com](https://deepinfra.com/pricing)
Fireworks   | $0.20                      | $0.50                        | [fireworks.ai](https://fireworks.ai/pricing)
Groq        | $0.08                      | $0.24                        | [groq.com](https://wow.groq.com/)
Lepton AI   | $0.07                      | $0.50                        | [lepton.ai](https://www.lepton.ai/pricing)
Mistral AI  | -                          | $0.70                        | [mistral.ai](https://mistral.ai/technology/#pricing)
Novita AI   | $0.06                      | -                            | [novita.ai](https://novita.ai/model-api/pricing)
OpenPipe    | $0.45                      | $1.40                        | [openpipe.ai](https://openpipe.ai/pricing#rates)
Together AI | $0.20                      | -                            | [together.ai](https://www.together.ai/pricing)

Note the above table above is not supposed to be an exhaustive comparison of such providers, it's missing important
information such as context size, inference speed, and model quantization.

## Checking out Groq

![](/images/llm-inference-as-a-service/groq-chip.png)

Groq was by far the choice that piqued my interest the most: it's unbelievably cheap and fast.

Groq was founded in 2016 by former Google employees who worked on Google’s Tensor Processing Unit (TPU). Leveraging their expertise in custom-designed chips for deep learning, they are now creating their own hardware, specifically what they call the "Language Processing Unit" (LPU):

> LPU Inference Engines are designed to overcome the two bottlenecks for LLMs–the amount of compute and memory bandwidth. An LPU system has as much or more compute as a Graphics Processor (GPU) and reduces the amount of time per word calculated, allowing faster generation of text sequences. With no external memory bandwidth bottlenecks an LPU Inference Engine delivers orders of magnitude better performance than Graphics Processor.

Installing their official package:

```
pip install groq
```

Testing it out:

```py
from groq import Groq

prompt = 'Tell me a joke'

groq_client = Groq(
    api_key='GROQ_SECRET_GOES_HERE',
)

chat_completion = groq_client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': prompt,
        }
    ],
    model='llama3-8b-8192',
)

print(f'Prompt time: {chat_completion.usage.prompt_time}')
print(f'Prompt tokens: {chat_completion.usage.prompt_tokens}')
print(f'Completion time: {chat_completion.usage.completion_time}')
print(f'Completion tokens: {chat_completion.usage.completion_tokens}')
print(f'Model: {chat_completion.model}')
print(f'Prompt: {prompt}')
print(f'Response: {chat_completion.choices[0].message.content}')
```

Output:

```
Prompt time: 0.003391819
Prompt tokens: 14
Completion time: 0.013561327
Completion tokens: 18
Model: llama3-8b-8192
Prompt: Tell me a joke
Response: Why couldn't the bicycle stand up by itself?

Because it was two-tired!
```

You can grab your API key [here](https://console.groq.com/keys) and check your usage
[here](https://console.groq.com/settings/usage).

## Comparing providers

Due how early and disruptive LLMs are (and how high inflation is nowadays), the pricing comparison table of this
article will be quickly outdated. [LLM Explorer](https://llm.extractum.io/gpu-hostings/) is a good resource on finding
more providers, and [OpenRouter](https://openrouter.ai/models/meta-llama/llama-3-8b-instruct) not only allows you to
quickly compare prices but also offers an API for [routing models and providers](https://openrouter.ai/docs/provider-routing)
on the fly.

![](/images/llm-inference-as-a-service/llm-explorer.png)

![](/images/llm-inference-as-a-service/open-router.png)

## Sources

- [The Groq LPU Inference Engine](https://wow.groq.com/lpu-inference-engine/)
- [groq/groq-python: The official Python Library for the Groq API](https://github.com/groq/groq-python)
- [LLM Explorer](https://llm.extractum.io/)
- [OpenRouter](https://openrouter.ai/)
