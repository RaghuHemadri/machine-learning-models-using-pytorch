import torch
from torch import nn, optim
from torch.autograd import Variable
 
class VRNNCell(nn.Module):
    def __init__(self):
        super(VRNNCell,self).__init__()
        self.phi_x = nn.Sequential(nn.Embedding(128,64), nn.Linear(64,64), nn.ELU())
        self.encoder = nn.Linear(128,64*2) # output hyperparameters
        self.phi_z = nn.Sequential(nn.Linear(64,64), nn.ELU())
        self.decoder = nn.Linear(128,128) # logits
        self.prior = nn.Linear(64,64*2) # output hyperparameters
        self.rnn = nn.GRUCell(128,64)
    def forward(self, x, hidden):
        x = self.phi_x(x)
        # 1. h =&amp;amp;amp;amp;amp;gt; z
        z_prior = self.prior(hidden)
        # 2. x + h =&amp;amp;amp;amp;amp;gt; z
        z_infer = self.encoder(torch.cat([x,hidden], dim=1))
        # sampling
        z = Variable(torch.randn(x.size(0),64))*z_infer[:,64:].exp()+z_infer[:,:64]
        z = self.phi_z(z)
        # 3. h + z =&amp;amp;amp;amp;amp;gt; x
        x_out = self.decoder(torch.cat([hidden, z], dim=1))
        # 4. x + z =&amp;amp;amp;amp;amp;gt; h
        hidden_next = self.rnn(torch.cat([x,z], dim=1),hidden)
        return x_out, hidden_next, z_prior, z_infer
    def calculate_loss(self, x, hidden):
        x_out, hidden_next, z_prior, z_infer = self.forward(x, hidden)
        # 1. logistic regression loss
        loss1 = nn.functional.cross_entropy(x_out, x)
        # 2. KL Divergence between Multivariate Gaussian
        mu_infer, log_sigma_infer = z_infer[:,:64], z_infer[:,64:]
        mu_prior, log_sigma_prior = z_prior[:,:64], z_prior[:,64:]
        loss2 = (2*(log_sigma_infer-log_sigma_prior)).exp() \
                + ((mu_infer-mu_prior)/log_sigma_prior.exp())**2 \
                - 2*(log_sigma_infer-log_sigma_prior) - 1
        loss2 = 0.5*loss2.sum(dim=1).mean()
        return loss1, loss2, hidden_next