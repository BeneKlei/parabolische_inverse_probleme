import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, inp, weights):
        # (batch, in_channels, timesteps), (in_channel, out_channel, timesteps) -> (batch, out_channel, timesteps)
        return torch.einsum('bix,iox->box', inp, weights)

    def forward(self, x):
        # (batch, in_channels bzw. width, timesteps)
        batchsize = x.shape[0]

        # TODO hat Padding hier Einfluss?
        x_ft = torch.fft.rfft(x)  # produces (batch, width, timesteps/2+1)

        # Placeholder for multiplication
        # produces shape (batchsize, out_channel, timesteps/2 + 1
        # note that // rounds down
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)

        # Multiply relevant Fourier modes
        # (batch, width, modes) * (width, width, modes) -> (batch, width, modes)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d_new(nn.Module):
    def __init__(self, 
                 dim_Q : int, 
                 dim_V : int,
                 modes : int, 
                 lifting_width : int):
    
        super(FNO1d_new, self).__init__()

        self.modes1 = modes
        self.lifting_width = lifting_width

        self.linear_p = nn.Linear(in_features=dim_Q, 
                                  out_features=self.lifting_width)

        self.width = self.lifting_width
        self.spect0 = SpectralConv1d(self.width, self.width, self.modes1)  # produces (batch, width, timesteps)
        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)  # Calculated per time step
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)  # produces (batch, width, timesteps)

        self.linear_q = nn.Linear(self.width, dim_V)
        self.output_layer = nn.Linear(dim_V, dim_V)

        self.activation = torch.nn.GELU()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, 
                func_inp : torch.Tensor
            ):

        # Concatenate processed inputs
        x1 = func_inp

        # FUNC INP LIFTING
        # shape (batch, timesteps, func inp features)
        x1 = self.linear_layer(x1, self.linear_p)  # produces (batch, timesteps, width)
        x1 = x1.permute(0, 2, 1)  # produces (batch, width, timesteps)

        x1 = self.fourier_layer(x1, self.spect0, self.lin0)    # Shape (batch_size, width, timesteps)

        x1 = self.fourier_layer(x1, self.spect1, self.lin1)

        x1 = self.fourier_layer(x1, self.spect2, self.lin2)

        # x1 = x1[..., :-self.padding]  # pad the domain if input is non-periodic
        x1 = x1.permute(0, 2, 1)

        x1 = self.linear_layer(x1, self.linear_q)
        x1 = self.output_layer(x1)

        return x1



