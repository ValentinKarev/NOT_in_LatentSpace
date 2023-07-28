## Imaging  library
from PIL import Image
from torchvision import transforms as tfms
import torch

## Basic libraries
import numpy as np
import matplotlib.pyplot as plt

## Loading a VAE model
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16, revision="fp16").to("cuda")

def load_image(p):
    '''
    Function to load images from a defined path
    '''
    return Image.open(p).convert('RGB').resize((512,512))

def pil_to_latents(image):
    '''
    Function to convert image to latents
    '''
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16) 
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist.detach().cpu()

def latents_to_pil(latents):
    '''
    Function to convert latents to images
    '''
    latents = latents.to(device="cuda", dtype=torch.float16)
    
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def latent_to_pil_from_bath_of_tensor(latents):
    latents = latents.to(device="cuda", dtype=torch.float16)
    
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    
    x_tmp = []
    
    for x in pil_images:
        #print(x)
        x = tfms.ToTensor()(x)
        #print(x)
        x_tmp.append(x)
        
    X = torch.stack(x_tmp)
    
    #print(X.shape)
    
    return X