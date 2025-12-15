# """
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Disp_Net(nn.Module):
    def __init__(self, fourier_dim=4096, output_dim=3):
        super(Disp_Net, self).__init__()
        
        self.fourier_freq = nn.Parameter(torch.empty(3, fourier_dim),requires_grad=False)
        self.fourier_bias = nn.Parameter(torch.ones([1, fourier_dim]),requires_grad=False)
        
        self.mlp = nn.Linear(fourier_dim, output_dim, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        high_band = 35
        low_band = 0.0
        c_y, c_x, c_z = np.meshgrid(
            np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
            np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
            np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
            indexing='ij'
        )
        fourier_init = np.stack((c_y.reshape([-1]), c_x.reshape([-1]), c_z.reshape([-1])), axis=0)
        
        self.fourier_freq.copy_(torch.tensor(fourier_init))
        nn.init.zeros_(self.mlp.weight)
            

    def forward(self, coord):
        fourier_feat = torch.matmul(coord, self.fourier_freq) + self.fourier_bias
        fourier_feat = torch.sin(fourier_feat)
        displacement = self.mlp(fourier_feat)
        
        return displacement
    
class Dens_Net(nn.Module):
    def __init__(self, fourier_dim=4096, output_dim=1):
        super(Dens_Net, self).__init__()

        self.fourier_freq = nn.Parameter(torch.empty(3, fourier_dim),requires_grad=False)
        self.fourier_bias = nn.Parameter(torch.ones([1, fourier_dim]),requires_grad=False)
        
        self.mlp = nn.Linear(fourier_dim, output_dim, bias=False)
        
        self._initialize_weights()

    def _initialize_weights(self):
        high_band = 35
        low_band = 0.0
        c_y, c_x, c_z = np.meshgrid(
            np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
            np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
            np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
            indexing='ij'
        )
        fourier_init = np.stack((c_y.reshape([-1]), c_x.reshape([-1]), c_z.reshape([-1])), axis=0)
        
        self.fourier_freq.copy_(torch.tensor(fourier_init))
        nn.init.zeros_(self.mlp.weight)

    def heaviside_projection(self, rho, beta):
        return (torch.tanh(beta * (rho - 0.5)) + 1.0) * 0.5
            

    def forward(self, coord, epoch_ratio):
        fourier_feat = torch.matmul(coord, self.fourier_freq) + self.fourier_bias
        fourier_feat = torch.sin(fourier_feat)
        rho = torch.sigmoid(self.mlp(fourier_feat))

        # beta = 1.0 + 29.0 * (1.0 - np.exp(-5.0 * epoch_ratio))

        # if epoch_ratio > 0.85:
        #     rho = self.heaviside_projection(rho, beta)
        # elif epoch_ratio > 0.35:
        #     t = (epoch_ratio - 0.35) / 0.5
        #     rho_hard = self.heaviside_projection(rho, beta)
        #     rho = (1.0 - t) * rho + t * rho_hard
        
        return rho

# """


"""
import numpy as np
import torch 
import torch.nn as nn

class Disp_Net(nn.Module):
    def __init__(self):
        super(Disp_Net, self).__init__()
        low_band = 0.0
        high_band = 35
        c_y, c_x, c_z=np.meshgrid(np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                  np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                  np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),indexing='ij')
        dlInit = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 0)
        self.register_parameter('kernel1', nn.Parameter(torch.tensor(dlInit, dtype=torch.float32, device=torch.device('cuda:0')), requires_grad=True))
        self.register_parameter('weights1', nn.Parameter(torch.zeros([dlInit.shape[1], 3], dtype=torch.float32, device=torch.device('cuda:0')), requires_grad=True))


    def __call__(self, coord):
        layer1 = torch.sin(torch.matmul(coord, self.kernel1) + torch.ones([1,self.kernel1.shape[1]], device=self.kernel1.device))
        u = torch.matmul(layer1, self.weights1)
        return u   
              
    def get_weights(self):
        return [self.weights1]

class Dens_Net(nn.Module):
    def __init__(self):
        super(Dens_Net, self).__init__()
        low_band = 0.0
        high_band = 35
        c_y, c_x, c_z=np.meshgrid(np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                  np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                  np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),indexing='ij')
        dlInit = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 0)
        # dlInit = torch.tensor(dlInit, dtype=torch.float32)
        # self.kernel1 = nn.Parameter(dlInit, requires_grad=True).to('cuda:0')
        # self.weights1 = nn.Parameter(torch.zeros([dlInit.shape[1],1]), requires_grad=True).to('cuda:0')
        self.register_parameter('kernel1', nn.Parameter(torch.tensor(dlInit, dtype=torch.float32, device=torch.device('cuda:0')), requires_grad=True))
        self.register_parameter('weights1', nn.Parameter(torch.zeros([dlInit.shape[1], 1], dtype=torch.float32, device=torch.device('cuda:0')), requires_grad=True))

    def __call__(self,coord):
        layer1 = torch.sin(torch.matmul(coord, 1.0 * self.kernel1) + torch.ones([1,self.kernel1.shape[1]], device=self.kernel1.device))
        rho = torch.sigmoid(torch.matmul(layer1, self.weights1))
        return rho
    
    def get_weights(self):
        return [self.weights1]
"""