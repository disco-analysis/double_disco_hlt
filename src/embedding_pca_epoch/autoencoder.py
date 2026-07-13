import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#encoder
class Encoder(nn.Module):
    def __init__(self, features, hidden_nodes, latent_dim, zero_init_logvar=True):
        super().__init__()
        layers = []
        input_dim = features

        #given hidden nodes, make the hidden layers
        for i, node in enumerate(hidden_nodes):
            layers += [
                #fully connected layer
                nn.Linear(input_dim, node),
                #activation function
                nn.ReLU(inplace=True),
            ]

            #output of this layer will be the input for the next
            input_dim = node
        self.backbone = nn.Sequential(*layers)

        #directly map to latent vector
        self.fc_latent = nn.Linear(input_dim, latent_dim)

    #forward pass through the encoder
    def forward(self, x):
        h = self.backbone(x)
        z = self.fc_latent(h)
        return z


#decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_nodes):
        super().__init__()
        assert len(hidden_nodes) >= 1, "decoder needs at least one layer (the output layer)."

        layers = []
        in_dim = latent_dim
        for i, node in enumerate(hidden_nodes):
            linear = nn.Linear(in_dim, node)
            
            #last layer uses small uniform init to prevent blowup (match diptarko code)
            if i == len(hidden_nodes) - 1:
                nn.init.uniform_(linear.weight, -0.05, 0.05)
                nn.init.uniform_(linear.bias, -0.05, 0.05)
            layers.append(linear)

            #apply batch norm + ReLU for all but last layer
            if i != len(hidden_nodes) - 1:
                layers.append(nn.BatchNorm1d(node))
                layers.append(nn.ReLU(inplace=True))

            in_dim = node

        self.net = nn.Sequential(*layers)

    #decodes z back to reconstructed features
    def forward(self, z):
        return self.net(z)


#vae class
class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        #scaling weight for reco loss
        self.alpha = float(config["alpha"])
        features = config["features"]
        latent_dim  = config["latent_dim"]
        enc_nodes = config["encoder_config"]["nodes"]
        dec_nodes = config["decoder_config"]["nodes"]

        #build encoder and decoder
        self.encoder = Encoder(features, enc_nodes, latent_dim)
        self.decoder = Decoder(latent_dim, dec_nodes)

    #forward pass through VAE
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    #computes all losses
    def compute_losses(self, x, target, reduction="mean", recon_loss_fn=None):
        recon, _ = self.forward(x)

        #reco loss
        if recon_loss_fn is None:
            #MSE summed over features per sample (if not using custom cyl_ptpz_mae loss)
            recon_per_elem = (recon - target) ** 2
            recon_per_sample = torch.sum(recon_per_elem, dim=1)
        else:
            #use the custom loss
            recon_per_sample = recon_loss_fn(recon, target)

        #apply scaling
        recon_loss = self.alpha*recon_per_sample

        if reduction == "mean":
            return recon_loss.mean()
        elif reduction == "sum":
            return recon_loss.sum()
        else:
            #return per-sample terms
            return recon_loss
