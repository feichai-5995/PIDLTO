import torch
from torch import nn
import numpy as np

import torch.autograd as autograd
import torch.nn.functional as F



class Pinn(nn.Module):
    def __init__(self, FEA_input, disp_model, params):
        super(Pinn,self).__init__()
        self.FEA_input = FEA_input
        self.disp_model = disp_model
                
        self.vf = params.vf
        self.penal = params.penal
        self.E0 = params.E0 
        self.nu = params.nu 
        self.keep_shell = params.keep_shell

        self.dlX_dx = self.FEA_input.dlX_dx
        self.dlX_dy = self.FEA_input.dlX_dy
        self.dlX_dz = self.FEA_input.dlX_dz

        self.device = torch.device('cuda:0')
        u_dlX = torch.tensor([0])
        u_dlX, scale_dlX= self.FEA_input.analytical_fixed_BC_disp(u_dlX, self.FEA_input.em_dlX)
        scale_dlX_mask = (scale_dlX == 0)
        self.N_fix = scale_dlX_mask.sum().detach()
        self.N_fix_ratio = self.N_fix / self.FEA_input.voxel_Ne

    def compute_gradients(self, output, input):
        return torch.autograd.grad(
            outputs=output, 
            inputs=input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )
    
    def flatten_params(self, params):
        return torch.cat([p.reshape(-1) for p in params])
    
    def compute_jacobian_matrix_exact(self, model, coords):
        model.zero_grad()
        coords = coords.detach().requires_grad_(True)
        out = model(coords)
        N, m = out.shape

        params = [p for p in model.parameters() if p.requires_grad]

        grads = []
        for i in range(N):
            grads_comp = []
            for j in range(m):
                g = autograd.grad(out[i, j], params, retain_graph=True, create_graph=False)
                grads_comp.append(torch.cat([gi.reshape(-1) for gi in g]))
            grads.append(torch.cat(grads_comp))
        return torch.stack(grads, dim=0)  # [N, P]
    
    def compute_ntk_from_jacobians(self, J1, J2=None):
        if J2 is None: J2 = J1
        return J1 @ J2.transpose(0, 1)

    def hutchinson_trace_estimate(self, model, coords, num_samples=20):
        model.zero_grad()
        coords = coords.detach().requires_grad_(True)
        outputs = model(coords)
        N, m = outputs.shape
        params = [p for p in model.parameters() if p.requires_grad]
        tr_est = 0.0

        for _ in range(num_samples):
            v = torch.randn((N, m),device=coords.device)
            JTv = autograd.grad(outputs, params, grad_outputs=v, retain_graph=True, create_graph=False)
            vjp = torch.cat([gi.reshape(-1) for gi in JTv])
            tr_est += (vjp @ vjp).item()
        return tr_est / num_samples
    
    def compute_ntk_weights(self, model, coord_u, coord_r, method='hutchinson'):
        if method == 'exact':
            Ju = self.compute_jacobian_matrix_exact(model, coord_u)
            Jr = self.compute_jacobian_matrix_exact(model, coord_r)
            Kuu = self.compute_ntk_from_jacobians(Ju)
            Krr = self.compute_ntk_from_jacobians(Jr)
            tr_uu, tr_rr = torch.trace(Kuu).item(), torch.trace(Krr).item()
        else:
            tr_uu = self.hutchinson_trace_estimate(model, coord_u, num_samples=50)
            tr_rr = self.hutchinson_trace_estimate(model, coord_r, num_samples=50)

        eps = 1e-12
        λ_fix = tr_rr / (tr_uu + eps)  
        λ_fix = max(min(λ_fix, 1e3), 1e-3)
        # λ_fix = λ_fix ** 0.5 
        if not hasattr(self, '_λ_fix_ema'):
            self._λ_fix_ema = λ_fix
        else:
            self._λ_fix_ema = 0.9 * self._λ_fix_ema + 0.1 * λ_fix

        # return self._λ_fix_ema
        return λ_fix

    def pinn_init_loss(self, xPhys_m, coord, disp_epoch_ratio):
        u = self.disp_model(coord)

        u, fixed_scale = self.FEA_input.analytical_fixed_BC_disp(u, coord)

###########################

        # u = (1 - alpha * fixed_scale.unsqueeze(1)) * u_raw
        ratio = max(0, 1 - (disp_epoch_ratio + 0.1) * 3.0)
        u = u * (fixed_scale.unsqueeze(1) * ratio + (1 - ratio))

        # fixed_mask = fixed_scale < 0.05 
        # u_fix = u * fixed_mask.unsqueeze(1).float() 
        # k_fix = self.E0 * 100  
        # E_fix = 0.5 * k_fix * torch.mean((u_fix ** 2).sum(dim=1, keepdim=True))
        # # E_fix = torch.mean((xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True))
        # # E_fix = (xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        # # E_fix = torch.mean(E_fix[fixed_mask])
        # λ_fix = disp_epoch_ratio  

        fixed_mask = (fixed_scale == 0) 
        # '''
        u_fix = u * fixed_mask.unsqueeze(1).float() 
        
        a = fixed_mask.sum()
        # N_fix_ratio = (fixed_mask.sum() / coord.shape[0]).detach()

        # u_fix = u * (1 - fixed_scale).unsqueeze(1)
        # weight = (1 - fixed_scale) ** 2
        # k_fix = self.E0 * min(1.0, disp_epoch_ratio * 1.5)
        # E_fix = (xPhys_m ** (3.0 * min(3.0, 2.0 + 2.0 * disp_epoch_ratio))) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        k_fix = self.E0 * 100
        E_fix = (xPhys_m ** self.penal) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        
        # E_fix = torch.mean(E_fix[fixed_mask])
        # E_fix = torch.sum(E_fix[fixed_mask])
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12)
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12) * self.N_fix
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12) * self.N_fix_ratio * self.FEA_input.em_V
        E_fix = torch.mean(E_fix[fixed_mask]) * self.N_fix_ratio * self.FEA_input.em_V

        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12)
        # E_fix = E_fix * self.N_fix

        # E_fix_soft = torch.mean(E_fix[fixed_mask])
        # u_penalty = torch.mean(u_fix**2)
        # '''
        λ_fix = self.compute_ntk_weights(self.disp_model, coord[fixed_mask], coord[~fixed_mask], method='hutchinson')
###########################
        
        u1 = u[:,0:1]
        u0 = u[:,1:2]
        u2 = u[:,2:3]
        
        ux_xyz = self.compute_gradients(u0, coord)[0]  # [∂u_x/∂y, ∂u_x/∂x, ∂u_x/∂z]
        uy_xyz = self.compute_gradients(u1, coord)[0]  # [∂u_y/∂y, ∂u_y/∂x, ∂u_y/∂z]
        uz_xyz = self.compute_gradients(u2, coord)[0]  # [∂u_z/∂y, ∂u_z/∂x, ∂u_z/∂z]
        
        eps11 = ux_xyz[:,1]
        eps12 = 0.5 * ux_xyz[:,0] + 0.5 * uy_xyz[:,1]
        eps13 = 0.5 * ux_xyz[:,2] + 0.5 * uz_xyz[:,1]
        eps22 = uy_xyz[:,0]
        eps23 = 0.5 * uy_xyz[:,2] + 0.5 * uz_xyz[:,0]
        eps33 = uz_xyz[:,2]


        lame_mu = self.E0 / (2.0 * (1.0 + self.nu))
        lame_lambda = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        trace_strain = eps11 + eps22 + eps33
        squared_diagonal = eps11 * eps11 + eps33 * eps33 + eps22 * eps22

        energy = 0.5 * lame_lambda * trace_strain * trace_strain + lame_mu * \
            (squared_diagonal + 2.0 * eps12 * eps12 +
            2.0 * eps13 * eps13 + 2.0 * eps23 * eps23)
        
        energy = torch.reshape(energy,[-1,1])*torch.pow(xPhys_m, self.penal).to('cuda:0')

        # E_fix = torch.mean(energy[fixed_mask]) * self.N_fix_ratio * self.FEA_input.em_V

        # energy_c = energy * self.FEA_input.voxel_Nm ** 3
        energy_c = energy * 2

        # \left(\mu\varepsilon_{i}:\varepsilon_{i}+\frac{\lambda\left(\operatorname{trace}\left(\varepsilon_{i}\right)\right)^{2}}{2}\right)

        energy_ans = self.FEA_input.em_V*torch.mean(energy)
        # energy_ans = torch.mean(energy)

        # zero_indices = [i for i, val in enumerate(self.FEA_input.F_vector.cpu().numpy()) if val == 0]
        # if zero_indices == [0,1]: em_A = self.dlX_dy * self.dlX_dx
        # elif zero_indices == [0,2]: em_A = self.dlX_dy * self.dlX_dz
        # elif zero_indices == [1,2]: em_A = self.dlX_dx * self.dlX_dz
        # force_l = em_A * torch.mean(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        force_l = torch.sum(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))


        # mean_energy = energy_ans.detach().abs() + 1e-12
        # mean_Efix = (E_fix).detach().abs() + 1e-12
        # alpha = 0.05 # desired fraction: E_fix ~ alpha * energy_ans
        # scale_factor = float((mean_energy / mean_Efix) * alpha)
        # # clamp scale_factor to reasonable range
        # scale_factor = max(min(scale_factor, 1e3), 1e-3)
       

        # alpha = min(1.0, disp_epoch_ratio * 2.0 + 0.5)
 
        # loss = energy_ans - force_l
        # loss = (energy_ans - force_l) + E_fix
        # u_fix_L2 = torch.mean(u_fix[fixed_mask]**2)
        # loss = (energy_ans - force_l) + λ_fix * u_fix_L2
        loss = (energy_ans - force_l) + λ_fix * E_fix
        # loss = (energy_ans - force_l) + λ_fix * (E_fix_soft + 50 * u_penalty)

        # compliance = -2 * energy_c

        return loss, energy_c, u

    def pinn_train_loss(self, xPhys_m, coord, disp_epoch_ratio):

        u = self.disp_model(coord)

        u, fixed_scale = self.FEA_input.analytical_fixed_BC_disp(u, coord)

        # if dens_epoch_ratio > 0.2:
        #      smooth_loss = self.disp_filter(u, fixed_scale, dens_epoch_ratio, xPhys_m, coord, origin_coord=None)

###########################

        # fixed_mask = fixed_scale < 0.05 
        # u_fix = u * fixed_mask.unsqueeze(1).float() 
        # k_fix = self.E0 * 100 
        # # E_fix = 0.5 * k_fix * torch.mean((u_fix ** 2).sum(dim=1, keepdim=True))
        # # E_fix = torch.mean((xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True))
        # E_fix = (xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        # E_fix = torch.mean(E_fix[fixed_mask])
        # λ_fix = 0.9 + (disp_epoch_ratio * 0.1) 

        # fixed_mask = fixed_scale < 0.05 
        # u_fix = u * fixed_mask.unsqueeze(1).float() 
        # k_fix = self.E0
        # E_fix = (xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        # E_fix = torch.mean(E_fix[fixed_mask])

        fixed_mask = (fixed_scale == 0)
        # '''
        u_fix = u * fixed_mask.unsqueeze(1).float() 
        # u_fix = u * (1 - fixed_scale).unsqueeze(1)
        k_fix = self.E0 * 100
        E_fix = (xPhys_m ** self.penal) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)

        # N_fix_ratio = (fixed_mask.sum() / coord.shape[0]).detach()

        # E_fix = torch.mean(E_fix[fixed_mask])
        # E_fix = torch.sum(E_fix[fixed_mask])
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12)
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12) * self.N_fix
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12) * self.N_fix_ratio * self.FEA_input.em_V
        E_fix = torch.mean(E_fix[fixed_mask]) * self.N_fix_ratio * self.FEA_input.em_V

        # u = u * fixed_scale.detach().unsqueeze(1)
        
        # u_raw = u.clone() 
        # alpha = 0.5
        # # u = (1 - alpha * fixed_scale.unsqueeze(1)) * u_raw
        # u = u_raw * ( fixed_scale.unsqueeze(1) + (1 - fixed_scale).unsqueeze(1) * (1 - alpha))

        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12)
        # E_fix = E_fix * self.N_fix
        # E_fix_soft = torch.mean(E_fix[fixed_mask])
        # u_penalty = torch.mean(u_fix**2)
        # '''

        λ_fix = self.compute_ntk_weights(self.disp_model, coord[fixed_mask], coord[~fixed_mask], method='hutchinson')

###########################

        u1 = u[:,0:1]
        u0 = u[:,1:2]
        u2 = u[:,2:3]
        
        ux_xyz = self.compute_gradients(u0, coord)[0]  # [∂u_x/∂y, ∂u_x/∂x, ∂u_x/∂z]
        uy_xyz = self.compute_gradients(u1, coord)[0]  # [∂u_y/∂y, ∂u_y/∂x, ∂u_y/∂z]
        uz_xyz = self.compute_gradients(u2, coord)[0]  # [∂u_z/∂y, ∂u_z/∂x, ∂u_z/∂z]
        
        eps11 = ux_xyz[:,1]
        eps12 = 0.5 * ux_xyz[:,0] + 0.5 * uy_xyz[:,1]
        eps13 = 0.5 * ux_xyz[:,2] + 0.5 * uz_xyz[:,1]
        eps22 = uy_xyz[:,0]
        eps23 = 0.5 * uy_xyz[:,2] + 0.5 * uz_xyz[:,0]
        eps33 = uz_xyz[:,2]


        lame_mu = self.E0 / (2.0 * (1.0 + self.nu))
        lame_lambda = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


        trace_strain = eps11 + eps22 + eps33
        squared_diagonal = eps11 * eps11 + eps33 * eps33 + eps22 * eps22
        energy = 0.5 * lame_lambda * trace_strain * trace_strain + lame_mu * \
            (squared_diagonal + 2.0 * eps12 * eps12 +
            2.0 * eps13 * eps13 + 2.0 * eps23 * eps23)
        

        energy = torch.reshape(energy,[-1,1])*torch.pow(xPhys_m, self.penal).to('cuda:0')

        # E_fix = torch.mean(energy[fixed_mask]) * self.N_fix_ratio * self.FEA_input.em_V

        # energy_c = energy * self.FEA_input.voxel_Nm ** 3
        energy_c = energy * 2

        # \left(\mu\varepsilon_{i}:\varepsilon_{i}+\frac{\lambda\left(\operatorname{trace}\left(\varepsilon_{i}\right)\right)^{2}}{2}\right)

        energy_ans =self.FEA_input.em_V*torch.mean(energy)

        # zero_indices = [i for i, val in enumerate(self.FEA_input.F_vector.cpu().numpy()) if val == 0]
        # if zero_indices == [0,1]: em_A = self.dlX_dy * self.dlX_dx
        # elif zero_indices == [0,2]: em_A = self.dlX_dy * self.dlX_dz
        # elif zero_indices == [1,2]: em_A = self.dlX_dx * self.dlX_dz
        # force_l = em_A * torch.mean(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        
        # force_l = torch.mean(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        # force_l = torch.sum(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        force_l = torch.sum(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))

        # mean_energy = (energy_ans - force_l).detach().abs() + 1e-12
        # mean_Efix = (E_fix).detach().abs() + 1e-12
        # alpha = 0.05 # desired fraction: E_fix ~ alpha * energy_ans
        # scale_factor = float((mean_energy / mean_Efix) * alpha)
        # # clamp scale_factor to reasonable range
        # scale_factor = max(min(scale_factor, 1e3), 1e-3)
        # λ_fix *= scale_factor
        
        # loss = energy_ans - force_l
        loss = (energy_ans - force_l) + λ_fix * E_fix
        # u_fix_L2 = torch.mean(u_fix[fixed_mask]**2)
        # loss = (energy_ans - force_l) + λ_fix * u_fix_L2
        # loss = (energy_ans - force_l) + λ_fix * (E_fix_soft + 50 * u_penalty)

        # compliance = -2 * energy_c

        return loss, energy_c, u

    def pinn_predict_loss(self, xPhys_m, coord, disp_epoch_ratio):

        u = self.disp_model(coord)

        u, fixed_scale = self.FEA_input.analytical_fixed_BC_disp(u, coord)

        # if dens_epoch_ratio > 0.2:
        #      smooth_loss = self.disp_filter(u, fixed_scale, dens_epoch_ratio, xPhys_m, coord, origin_coord=None)

###########################

        # fixed_mask = fixed_scale < 0.05 
        # u_fix = u * fixed_mask.unsqueeze(1).float() 
        # k_fix = self.E0 * 100 
        # # E_fix = 0.5 * k_fix * torch.mean((u_fix ** 2).sum(dim=1, keepdim=True))
        # # E_fix = torch.mean((xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True))
        # E_fix = (xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        # E_fix = torch.mean(E_fix[fixed_mask])
        # λ_fix = 0.9 + (disp_epoch_ratio * 0.1) 

        # fixed_mask = fixed_scale < 0.05
        # u_fix = u * fixed_mask.unsqueeze(1).float() 
        # k_fix = self.E0
        # E_fix = (xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        # E_fix = torch.mean(E_fix[fixed_mask])

        # fixed_mask = (fixed_scale == 0) 
        # u_fix = u * fixed_mask.unsqueeze(1).float() 
        # # u_fix = u * (1 - fixed_scale).unsqueeze(1)
        # k_fix = self.E0 
        # E_fix = (xPhys_m ** 3.0) * 0.5 * k_fix * (u_fix ** 2).sum(dim=1, keepdim=True)
        # E_fix = torch.mean(E_fix[fixed_mask])
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12)
        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12) * self.N_fix
        # E_fix = torch.sum(E_fix[fixed_mask])

        # u = u * fixed_scale.detach().unsqueeze(1)
        
        # u_raw = u.clone() 
        # alpha = 0.5
        # # u = (1 - alpha * fixed_scale.unsqueeze(1)) * u_raw
        # u = u_raw * ( fixed_scale.unsqueeze(1) + (1 - fixed_scale).unsqueeze(1) * (1 - alpha))

        # E_fix = torch.sum(E_fix[fixed_mask]) / (fixed_mask.sum() + 1e-12)
        # E_fix = E_fix * self.N_fix
        # E_fix_soft = torch.mean(E_fix[fixed_mask])
        # u_penalty = torch.mean(u_fix**2)
        

        # λ_fix = self.compute_ntk_weights(self.disp_model, coord[fixed_mask], coord[~fixed_mask], method='hutchinson')

###########################

        fixed_mask = (fixed_scale != 0) 
        u = u * fixed_mask.unsqueeze(1).float() 

        u1 = u[:,0:1]
        u0 = u[:,1:2]
        u2 = u[:,2:3]
        
        ux_xyz = self.compute_gradients(u0, coord)[0]  # [∂u_x/∂y, ∂u_x/∂x, ∂u_x/∂z]
        uy_xyz = self.compute_gradients(u1, coord)[0]  # [∂u_y/∂y, ∂u_y/∂x, ∂u_y/∂z]
        uz_xyz = self.compute_gradients(u2, coord)[0]  # [∂u_z/∂y, ∂u_z/∂x, ∂u_z/∂z]
    
        eps11 = ux_xyz[:,1]
        eps12 = 0.5 * ux_xyz[:,0] + 0.5 * uy_xyz[:,1]
        eps13 = 0.5 * ux_xyz[:,2] + 0.5 * uz_xyz[:,1]
        eps22 = uy_xyz[:,0]
        eps23 = 0.5 * uy_xyz[:,2] + 0.5 * uz_xyz[:,0]
        eps33 = uz_xyz[:,2]


        lame_mu = self.E0 / (2.0 * (1.0 + self.nu))

        lame_lambda = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


        trace_strain = eps11 + eps22 + eps33
        squared_diagonal = eps11 * eps11 + eps33 * eps33 + eps22 * eps22
        energy = 0.5 * lame_lambda * trace_strain * trace_strain + lame_mu * \
            (squared_diagonal + 2.0 * eps12 * eps12 +
            2.0 * eps13 * eps13 + 2.0 * eps23 * eps23)



        energy = torch.reshape(energy,[-1,1])*torch.pow(xPhys_m, 3.0).to('cuda:0')
        # energy_c = energy * self.FEA_input.voxel_Nm ** 3
        energy_c = energy * 2

        # \left(\mu\varepsilon_{i}:\varepsilon_{i}+\frac{\lambda\left(\operatorname{trace}\left(\varepsilon_{i}\right)\right)^{2}}{2}\right)

        energy_ans =self.FEA_input.em_V*torch.mean(energy)

        # zero_indices = [i for i, val in enumerate(self.FEA_input.F_vector.cpu().numpy()) if val == 0]
        # if zero_indices == [0,1]: em_A = self.dlX_dy * self.dlX_dx
        # elif zero_indices == [0,2]: em_A = self.dlX_dy * self.dlX_dz
        # elif zero_indices == [1,2]: em_A = self.dlX_dx * self.dlX_dz
        # force_l = em_A * torch.mean(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        
        # force_l = torch.mean(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        # force_l = torch.sum(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        force_l = torch.sum(torch.matmul(self.disp_model(self.FEA_input.em_dlX_force),self.FEA_input.F_vector))
        
        # loss = energy_ans - force_l
        # loss = (energy_ans - force_l) + λ_fix * E_fix
        # loss = (energy_ans - force_l) + λ_fix * (E_fix_soft + 50 * u_penalty)
        # compliance = -2 * energy_c

        return energy_c


    def disp_filter(self, u, coord, origin_coord=None):

        from torch_cluster import radius_graph, knn_graph
        from torch_geometric.data import Data
        from torch_geometric.transforms import LargestConnectedComponents
        
        num_coord = coord.shape[0]

        if origin_coord is None:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.02
        else:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.5

        # print(f'coord num {coord_num}')
        edge_index = radius_graph(coord, r=r, loop=False, max_num_neighbors=64,)
        # print(f'edge num {edge_index.shape[1]}')  

        row, col = edge_index
        coord_i = coord[row]  
        coord_j = coord[col] 
        diffs = torch.abs(coord_i - coord_j)

        if origin_coord is None:
            thresholds = torch.tensor([self.dlX_dy*1.02, self.dlX_dx*1.02, self.dlX_dz*1.02], device=self.device)
            valid_edge_mask = (diffs <= thresholds).all(dim=1)
        else:
            origin_thresholds = torch.tensor([self.dlX_dy*1.02, self.dlX_dx*1.02, self.dlX_dz*1.02], device=self.device)
            origin_coord_i = origin_coord[row]  
            origin_coord_j = origin_coord[col] 
            origin_diffs = torch.abs(origin_coord_i - origin_coord_j)
            thresholds = torch.tensor([self.dlX_dy*1.5, self.dlX_dx*1.5, self.dlX_dz*1.5], device=self.device)
            valid_edge_mask = ((origin_diffs <= origin_thresholds) & (diffs <= thresholds)).all(dim=1)

        # print(f'valid edge num {valid_edge_mask.sum()}')
        edge_index_valid = edge_index[:,valid_edge_mask]
        connect_test_point_array = torch.arange(num_coord,device=self.device).view(-1, 1)
        data = Data(x=connect_test_point_array, edge_index=edge_index_valid, num_nodes=num_coord)

        transform = LargestConnectedComponents(num_components=1,connection='weak')

        transformed_data = transform(data)
        node_indices = transformed_data.x.squeeze().long()
        
        fill_indices_mask = torch.zeros(num_coord, dtype=torch.bool)
        fill_indices = node_indices

        fill_indices_mask[fill_indices] = True

        fill_u_each_point = torch.norm(u[fill_indices], dim=1, keepdim=False)
        fill_coord = coord[fill_indices]
        fill_edge_i, fill_edge_j = transformed_data.edge_index 
        fill_u_edge_i = fill_u_each_point[fill_edge_i]
        fill_u_edge_j = fill_u_each_point[fill_edge_j]
        # dir_mask = ((fill_u_edge_i >= fill_u_edge_j) | ((fill_u_edge_i - fill_u_edge_j).abs() < 1e-4)).flatten()
        fill_u_each_point_epsilon = fill_u_each_point.max() * 0.01
        dir_mask = ((fill_u_edge_i >= fill_u_edge_j) | ((fill_u_edge_i - fill_u_edge_j).abs() < fill_u_each_point_epsilon)).flatten()

        num_dir = dir_mask.sum()

        directed_edges = torch.stack([
            torch.where(dir_mask, fill_edge_i, fill_edge_j),
            torch.where(dir_mask, fill_edge_j, fill_edge_i)], dim=0)

        diff_fixed = (fill_coord[:, None, :] - self.FEA_input.em_dlX_fixed[None, :, :]).abs()
        match_fixed = (diff_fixed < 1e-8).all(dim=-1)
        fixed_mask = match_fixed.any(dim=1)

        diff_force = (fill_coord[:, None, :] - self.FEA_input.em_dlX_force[None, :, :]).abs()
        match_force = (diff_force < 1e-8).all(dim=-1)
        force_mask = match_force.any(dim=1)

        fixed_indices = fixed_mask.nonzero(as_tuple=False).squeeze(-1)
        force_indices = force_mask.nonzero(as_tuple=False).squeeze(-1)
        # fill_fixed = fill_coord[fixed_indices]
        # fill_force = fill_coord[force_indices]
        # 连通的固定点和受力点在连通点集中的索引

        num_nodes = fill_coord.shape[0]
        
        reachable_from_force = self.bfs_sparse_tensor_multi_source(directed_edges, force_indices, num_nodes, max_steps=None)
        num_reachable_force = reachable_from_force.sum()

        rev_edges = torch.stack([directed_edges[1], directed_edges[0]], dim=0)

        reachable_to_fixed = self.bfs_sparse_tensor_multi_source(rev_edges, fixed_indices, num_nodes, max_steps=None)
        num_reachable_fixed = reachable_to_fixed.sum()


        path_graph_mask = reachable_from_force & reachable_to_fixed
        unreachable_indices = (~path_graph_mask).nonzero(as_tuple=False).squeeze(-1)
        if unreachable_indices.numel() == 0:
            unreachable_indices = torch.empty((0,), dtype=torch.long, device=self.device)

        return unreachable_indices, path_graph_mask  


    def bfs_sparse_tensor_multi_source(self, edge_index, start_index, num_index, max_steps=None, batch_frontier=False):

        from torch_sparse import SparseTensor

        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_index, num_index))

        visited = torch.zeros(num_index, dtype=torch.bool, device=self.device)

        if start_index.numel() == 0:
            return visited

        visited[start_index] = True
        steps = 0
        max_steps = num_index if max_steps is None else int(max_steps)

        while steps < max_steps:
            sub = adj[start_index]
            neigh = sub.storage.col()
            if neigh.numel() == 0:
                break
            neigh = neigh.unique()
            new_nodes = neigh[~visited[neigh]]
            if new_nodes.numel() == 0:
                break
            visited[new_nodes] = True
            start_index = new_nodes
            steps += 1


            '''

            visited[neigh] = True
            new_nodes = ~visited
            if new_nodes.numel() == 0:
                break
            start_index = neigh
            steps += 1

            '''

        num_reachable = visited.sum()
        return visited
    