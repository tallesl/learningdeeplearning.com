---
title: Prompting Tips
date: 2024-08-02
---

Prompting LLMs for content generation or answers is here to stay, but its open-ended nature can be both a blessing and a curse. In this article, we’ll show three quick tips to improve your prompting experience, tailoring the output to your desire from the very first message sent to the model:
- **Use clear and direct language**: Even though LLMs are impressively 'human', remember that they are still machines and prefer clear and direct instructions.
- **Personas**: Specify who the model is supposed to impersonate and who the target audience of the content is.
- **Define the format and length**: Do you want a long article? A poem maybe? Bullet lists? Haiku? Make sure to mention the desired format and length.
- **External information**: Adding external information (such as citations) can greatly improve the result instead of relying solely on the LLM's previously acquired knowledge.
- **Iterate**: A good result is iterative

## Example #1: Technical article about AMD

Prompt:

```
You are a knowledgeable AI specialist writing to a non-technical audience about how AMD is fighting back NVIDIA
dominance in the field. Mention how competition is beneficial to consumers and cite examples how such technology can
aid anyone using a computer on day to day tasks.

Make sure to reference the following AMD official blog posts on your article:

"""
AMD has partnered with TensorStack, an AI technology company based out of New Zealand, to bring a painless, easy to use
and optimized end-user experience to AMD Ryzen™ AI 300 Series processors and Radeon™ RX 7000 Series graphics customers. 

Amuse 2.0.0 is an easy to install program, that has no dependencies, a simplified user interface called “Ez Mode” and
will quickly take care of all the configuration required to jump start your AI image generation journey. It also
features an AMD XDNA™ Super Resolution solution to speed up and increase output size by 2x at the end of the image
generation stage.

When starting up Amuse for the first time, you will be able to breeze through the installation process just like any
other software and should eventually arrive on the window shown below. If an AMD XDNA™ NPU was successfully detected by
the application, the AMD XDNA™ Super Resolution toggle will automatically be checked. As of right now, the only AMD
family of products officially supported by AMD XDNA™ Super Resolution are Ryzen AI™ 300 Series and Ryzen™ 8000 Series
(with the required OEM driver update) processors. 
"""

"""
Generative AI is changing the way software engineers work today. Did you know that you can build your own coding
Copilot just using the AMD RadeonTM graphic card locally? That’s right. AMD provides powerful large-model inference
acceleration capabilities through the latest and advanced AMD RDNATM architecture, that powers not just cutting-edge
gaming but also high-performance AI experience. With the help of the open software development platform AMD ROCmTM, now
it is possible for software developers to implement GPT-like code generation functions on the desktop machines. This
blog will share with you how to build your personal coding Copilot with Radeon graphic card, Continue (name of an
open-source integrated development environment, act as an extension of VSCode and JetBrains that enable developers to
create their own modular AI software development system easily), and LM Studio plus the latest open-source large model
Llama3.
"""

"""
For users that are looking to drive generative AI locally, AMD Radeon™ GPUs can harness the power of on-device AI
processing to unlock new experiences and gain access to personalized and real-time AI performance.

LLMs are no longer the preserve of big businesses with dedicated IT departments, running services in the cloud. With
the combined power of select AMD Radeon desktop GPUs and AMD ROCm software, new open-source LLMs like Meta's Llama 2
and 3 – including the just released Llama 3.1 – mean that even small businesses can run their own customized AI tools
locally, on standard desktop PCs or workstations, without the need to store sensitive data online4.

AMD AI desktop systems equipped with a Radeon PRO W7900 GPU running AMD ROCm 6.1 software and powered by Ryzen™
Threadripper™ PRO processors, represent a new client solution to fine-tune and run inference on LLMs with high
precision.

As well, AMD PCs equipped with DirectML supported AMD GPUs can also take advantage of running Llama3.1 locally on your
devices accelerated via DirectML AI frameworks optimized for AMD device.
"""

Your article should respect the following structure: a brief paragraph introducing the problem (100 words maximum), two
longer paragraphs elaborating (200 words each, maximum), a brief closing statement (100 words maximum) summarizing the
article's main ideas in bullet points.
```

ChatGPT 4o response:

```
AMD's Strategic Moves to Counter NVIDIA's Dominance

In the fiercely competitive landscape of graphics and AI processing technologies, AMD is making significant strides to
challenge NVIDIA's long-standing dominance. The battle between these two giants is not just a corporate rivalry; it has
profound implications for consumers and the broader tech ecosystem. As AMD introduces innovative technologies and
partnerships, consumers stand to benefit from enhanced competition which typically leads to more choice, better
performance, and lower prices.

AMD's partnership with TensorStack and the introduction of the Amuse 2.0.0 software exemplify how the company is
simplifying the integration of AI capabilities into everyday computing with their Ryzen™ AI 300 Series and Radeon™ RX
7000 Series. This move not only democratizes access to advanced AI technologies but also enhances user experience with
features like AMD XDNA™ Super Resolution, which effortlessly improves image generation speeds and quality. This level
of innovation is crucial, as it enables both casual users and professionals to leverage AI for creative and
productivity tasks without needing specialized knowledge.

Furthermore, AMD's efforts in making large model inference more accessible through its Radeon GPUs and ROCm software
platforms are transforming the software development landscape. By enabling the local execution of powerful AI models
like Llama 3.1 on standard desktops, AMD is breaking down barriers that once limited advanced AI tools to large
corporations with significant resources. This approach not only ensures data privacy by allowing local data processing
but also empowers small businesses and individual developers to customize AI tools according to their specific needs.
The ability to run these tools on easily accessible and affordable AMD-powered systems underscores a pivotal shift
towards more inclusive and versatile computing environments.

In summary, AMD's strategic initiatives are reshaping the technological landscape by:

• Enhancing consumer choice and driving down costs through vigorous competition.
• Democratizing access to cutting-edge AI technologies, making them accessible to a broader audience.
• Empowering businesses and individuals by localizing AI capabilities, enhancing privacy, and customization.

These efforts reflect a broader trend of technological empowerment that promises to redefine our interaction with
digital environments in our daily lives.
```

Llama 3 70B response:

```
The Dominance of NVIDIA in the Field of AI: How AMD is Fighting Back

For years, NVIDIA has been the undisputed leader in the field of artificial intelligence (AI) and graphics processing.
However, AMD is now fighting back, and this competition is beneficial to consumers. With AMD's innovative solutions,
users can enjoy faster, more efficient, and more affordable AI experiences.

One of the key areas where AMD is making a significant impact is in the field of generative AI. With the launch of its
Ryzen AI 300 Series processors and Radeon RX 7000 Series graphics, AMD has partnered with TensorStack to bring a
seamless and optimized end-user experience to its customers. The Amuse 2.0.0 program, which features an AMD XDNA Super
Resolution solution, allows users to easily install and configure their AI image generation journey. This means that
users can now generate high-quality images quickly and efficiently, without the need for complex configurations or
dependencies. Moreover, AMD's Radeon GPUs can harness the power of on-device AI processing, unlocking new experiences
and providing personalized and real-time AI performance.

The implications of AMD's innovations are far-reaching. For software engineers, AMD's Radeon GPUs can be used to build
their own coding Copilot, allowing them to implement GPT-like code generation functions on their desktop machines. This
means that developers can now create their own modular AI software development system easily, without the need for
cloud-based services. Moreover, AMD's AI desktop systems equipped with Radeon PRO W7900 GPUs and Ryzen Threadripper PRO
processors can fine-tune and run inference on large language models (LLMs) with high precision, making it possible for
small businesses to run their own customized AI tools locally.

In summary, AMD is fighting back against NVIDIA's dominance in the field of AI, and this competition is benefiting
consumers. With AMD's innovative solutions, users can enjoy faster, more efficient, and more affordable AI experiences.
Key takeaways include:
• AMD's Ryzen AI 300 Series processors and Radeon RX 7000 Series graphics provide a seamless and optimized end-user
experience for AI image generation.
• AMD's Radeon GPUs can be used to build coding Copilots and implement GPT-like code generation functions on desktop
machines.
• AMD's AI desktop systems can fine-tune and run inference on LLMs with high precision, making it possible for small
businesses to run their own customized AI tools locally.
```

## Example #2: Game advertisement for kids

Prompt:

```
You are a marketing specialist at Nintendo creating a short advertisement for the new "Super Mario Wonder" game. The ad
is targeted to children from 10 to 14 years old. Write it in a playful manner using words that little kids would
understand. Use first-person and write it like it’s Mario himself speaking. Make it two paragraphs no longer than 70
words each.

Here are some game features to consider when elaborating the text:

"""
New power-ups include a fruit that transforms the player into an elephant, a flower that allows the player to create
bubbles that capture enemies, and a mushroom that gives the player a drill hat that allows them to burrow into the
ground or ceiling to evade enemies or bypass obstacles.

Wonder introduces the Wonder Flower, which triggers bizarre effects such as pipes coming to life, hordes of enemies
spawning, and character appearances and abilities changing. The effects end when a player collects a Wonder Seed or
leaves the area of effect.
"""
```

ChatGPT 4o response:

```
Mamma mia! It's-a me, Mario! Get ready for an adventure like no other in Super Mario Wonder! Ever wanted to be an
elephant? Now you can with our magical fruit! Or create bubbles to catch those pesky enemies with the flower power-up.
And don't forget the drill hat mushroom—it lets you burrow underground or even into the ceiling!

But wait, there's more! Discover the amazing Wonder Flower that turns the world crazy! Pipes come to life, and tons of
enemies appear. It’s super wild! Collect a Wonder Seed to stop the madness or leave the area to escape. Join me and
let’s-a go on this super fun adventure!
```

Llama 3 70B response:

```
Heya, it's-a me, Mario! I'm super excited to introduce you to Super Mario Wonder! This new adventure is packed with
crazy cool power-ups that'll make you go "Whoa!" Imagine turning into an elephant and stomping on Goombas, or blowing
bubbles to trap Koopa Troopas. And with my new drill hat, I can dig my way out of trouble or sneak past pesky enemies.
It's-a gonna be a wild ride!

But wait, there's more! In Super Mario Wonder, I've got a special trick up my sleeve - the Wonder Flower! When I grab
it, the whole world goes wacky! Pipes start dancing, enemies go crazy, and even my friends and I change our looks and
powers! It's-a total chaos, but in a super fun way! So, are you ready to join me on this wacky adventure? Let's-a play
Super Mario Wonder and find out what wonders await us!
```

## Further reading

Those are the very basics TODO:

- Chain of Thought (CoT): A reasoning process where the model generates intermediate steps to solve a problem.
- n-shot Learning: A learning paradigm where the model is trained with a small number of examples for each class.
- RAG (Retrieval-Augmented Generation): A technique that combines retrieval of relevant documents with generative models to produce more accurate and informed responses.
- Fine-tuning: The process of training a pre-trained model on a specific task or dataset to improve its performance for that particular task.
- Context Size: The maximum amount of text or tokens that a model can consider at once when generating a response.
- Temperature: A parameter that controls the randomness of predictions by scaling the logits before applying the softmax function.

## Sources

- [Gemini for Google Workspace - Prompting guide 101](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)
- [Prompt engineering - OpenAI API](https://platform.openai.com/docs/guides/prompt-engineering)
- [Introducing Amuse 2.0 Beta with AMD XDNA Super Resolution: a fully local, AI experience](https://community.amd.com/t5/ai/introducing-amuse-2-0-beta-with-amd-xdna-super-resolution-a/ba-p/697374)
- [Build Your Local Coding Copilot with AMD Radeon GPU Platform](https://community.amd.com/t5/ai/build-your-local-coding-copilot-with-amd-radeon-gpu-platform/ba-p/689459)
- [Llama 3.1: Ready to Run on AMD platforms from data center, edge to AI PCs](https://community.amd.com/t5/ai/llama-3-1-ready-to-run-on-amd-platforms-from-data-center-edge-to/ba-p/697323)
- [Super Mario Bros. Wonder - Wikipedia](https://en.wikipedia.org/wiki/Super_Mario_Bros._Wonder)





