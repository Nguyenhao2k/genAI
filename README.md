# Description

D-Soft: Research and Implement Image-to-Video application

## Approach methods

1. Variational Autoencoder (VAE): Encode images to a compressed size, then decode back to the original size, while learning the distribution of the data

2. Generative Adversarial Network (GAN): They have two parts (the Generator and the Discriminator) that help each other get better. The Generator learns to make data that looks real, and the Discriminator learns to tell the difference between real and fake data.

3. Flow-based Generative Model: Create new data that’s similar to the data they were trained on and then calculate how likely a certain output is

4. Auto-Regressive Model: Model the conditional probability of each pixel given previous pixels. Then use the probability distribution to generate new data

5. [Diffusion Model:](https://arxiv.org/pdf/2006.11239.pdf) Systematically and slowly destroy struture in data distribution though an iterative ``forward diffusion process``. We then learn a ``reverse diffusion process`` that restores structure in data, yielding a highly flexible and tractable generative model of the data.

## SOTA models

1. [Stable video diffusion (SDM):](https://static1.squarespace.com/static/6213c340453c3f502425776e/t/655ce779b9d47d342a93c890/1700587395994/stable_video_diffusion.pdf)

2. [Laten flow diffusion model (LDM):](https://arxiv.org/pdf/2303.13744.pdf)

3. [Cascade diffusion model (CDM):](https://arxiv.org/pdf/2311.04145.pdf)

## Model Architecture (U-NET)

- Forward process:

1. Noise Scheduler
2. Neural Network
3. Timestep Encoding

## Fine-Tuning Pre-trained Model

### Applications

- DALL-E: [Link](https://openai.com/dall-e-3)

- Image GPT: [Link](https://openai.com/research/image-gpt)

- Mid Journey: [Link](https://www.midjourney.com/explore)

- Gen-2 Gen: [Link](https://app.runwayml.com/login)
