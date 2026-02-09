import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class StereoTrackEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super(StereoTrackEncoder, self).__init__()
        
        self.dropout_0 = nn.Dropout(
            p=dropout_rate, 
            inplace=False
        )
        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.activation_0 = nn.LeakyReLU(negative_slope=0.2)
        self.bn_0 = nn.BatchNorm1d(
            hidden_dim,
            eps = 1e-05,
            momentum = 0.1,
            affine = True,
            track_running_stats = True
        )
        self.dropout_1 = nn.Dropout(
            p=dropout_rate,
            inplace=False
        )
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.bn_1 = nn.BatchNorm1d(
            hidden_dim,
            eps = 1e-05,
            momentum = 0.1,
            affine = True,
            track_running_stats = True
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, adj):
        h_1 = self.linear_0(x)
        h_1 = self.bn_0(h_1)
        h_1 = self.activation_0(h_1)
        h_1 = self.dropout_0(h_1)

        h_2 = self.linear_1(h_1)
        h_2 = self.bn_1(h_2)
        h_2 = self.activation_1(h_2)
        h_2 = self.dropout_1(h_2)

        z_mean = self.mean(h_2)
        z_log_var = F.softplus(self.log_var(h_2))

        z_sample = D.Normal(z_mean, z_log_var)
        
        z_spatial = torch.mm(adj.T, z_mean)

        return z_sample, z_spatial, z_mean


class StereoTrackDecoder(nn.Module):
    def __init__(self, gene_embedding, var_index):
        super(StereoTrackDecoder, self).__init__()
        self.gene_embedding = gene_embedding
        self.var_index = var_index
        self.activation = nn.ReLU()

    def forward(self, z):
        x_recon = torch.mm(z, self.gene_embedding[:, self.var_index])
        x_recon = self.activation(x_recon)
        return x_recon


class StereoTrackModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate, 
                 var_name, all_genes, use_spatial_decoder=True):
        super(StereoTrackModel, self).__init__()
        
        self.encoder = StereoTrackEncoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        
        self.gene_embedding = nn.Parameter(torch.zeros(latent_dim, len(all_genes)))
        nn.init.xavier_uniform_(self.gene_embedding)
        
        self.var_index = [all_genes.index(g) for g in var_name]
        self.decoder = StereoTrackDecoder(self.gene_embedding, self.var_index)
        
        self.use_spatial_decoder = use_spatial_decoder

    def forward(self, x, adj):

        z_distribution, z_spatial, z_mean = self.encoder(x, adj)
        
        if self.use_spatial_decoder:
            x_recon = self.decoder(z_spatial)
        else:
            x_recon = self.decoder(z_mean)
        
        return x_recon, z_distribution, z_spatial, z_mean
