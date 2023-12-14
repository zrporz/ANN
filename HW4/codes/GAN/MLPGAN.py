import torch.nn as nn
import torch
import os
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def get_generator(num_channels, latent_dim, hidden_dim, device):
    model = GeneratorMLP(num_channels, latent_dim, hidden_dim).to(device)
    model.apply(weights_init)
    return model

def get_discriminator(num_channels, hidden_dim, device):
    model = DiscriminatorMLP(num_channels, hidden_dim).to(device)
    model.apply(weights_init)
    return model

class GeneratorMLP(nn.Module):
    def __init__(self, num_channels, latent_dim, hidden_dim):
        super(GeneratorMLP, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # MLP layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 32 * 32 * num_channels),  # Adjust the size to match the image size
            nn.Tanh()
        )

    def forward(self, z):
        x = self.decoder(z.view(z.size(0), -1))
        x = x.view(x.size(0), self.num_channels, 32, 32)  # Reshape to the image size
        return x
    
    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'generator.bin')):
                path = os.path.join(ckpt_dir, 'generator.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'generator.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'generator.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
    
class DiscriminatorMLP(nn.Module):
    def __init__(self, num_channels, hidden_dim):
        super(DiscriminatorMLP, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # MLP layers
        self.clf = nn.Sequential(
            nn.Linear(32 * 32 * num_channels, hidden_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.clf(x.view(x.size(0), -1) ).view(-1, 1).squeeze(1)
    
    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'discriminator.bin')):
                path = os.path.join(ckpt_dir, 'discriminator.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'discriminator.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'discriminator.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]