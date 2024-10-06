
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        
        # YOUR IMPLEMENTATION HERE!
        
        # Precompute and store the parameters for performing noise addition for a given timestep.
        betas = torch.linspace(beta_1, beta_T, T) # Linear schedule from beta_1 to beta_T
        
        # Get the betas in beta_1, beta_T range
        self.register_buffer('betas', betas)
        
        # Get alphas and cumprods of alphas
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1 - alpha_bar))

    def forward(self, x_0, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        Inputs  - Original images (batch_size x 3 x 32 x 32), class labels[1 to 10] (batch_size dimension) 
        Outputs - Loss value (mse works for this application)
        
        """
        
        # pick batched random timestep below self.T. (torch.Size([batch_size]))
        t = torch.randint(0, self.T, (x_0.shape[0],))
        
        # Generate random noise from normal distribution with 0 mean and 1 variance (torch.Size([batch_size, 3, 32, 32])
        noise = torch.randn_like(x_0)
        
        # Compute the x_t (images obtained after corrupting the input images by t times)  (torch.Size([batch_size, 3, 32, 32])
        mean = self.get_buffer('sqrt_alphas_bar')[t] * x_0
        std = self.get_buffer('sqrt_one_minus_alphas_bar')[t]
        x_t = mean + std * noise
        
        # Call your diffusion model to get the predict the noise -  t is a random index
        pred = self.model(x_t, t, labels)
        
        # Compute your loss for model prediction and ground truth noise (that you just generated)
        loss = F.mse_loss(pred, x_0)
    
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w


        # YOUR IMPLEMENTATION HERE!

        # Store any constant in register_buffer for quick usage in forward step
        betas = torch.linspace(beta_1, beta_T, T) # Linear schedule from beta_1 to beta_T
        
        # Get the betas in beta_1, beta_T range
        self.register_buffer('betas', betas)
        
        # Get alphas and cumprods of alphas
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('alphas', 1 - betas)
        # self.register_buffer('sqrt_alphas_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1 - alpha_bar))

    def forward(self, x_T, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        """
        
        alphas = self.get_buffer('alphas')
        # sqrt_alphas_bar = self.get_buffer('sqrt_alphas_bar')
        sqrt_one_minus_alphas_bar = self.get_buffer('sqrt_one_minus_alphas_bar')
        
        x_t = x_T
        for time_step in reversed(range(self.T)):

            # YOUR IMPLEMENTATION HERE!
            t = time_step + 1
            
            z = torch.randn_like(x_t) if time_step > 0 else 0
            a_t = alphas[t]
            
            x_t = 1/a_t * (x_t - ((1 - a_t) / sqrt_one_minus_alphas_bar) * self.model(x_t, t, labels) + sqrt_one_minus_alphas_bar * z)
            
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


