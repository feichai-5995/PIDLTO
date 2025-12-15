import torch
from torch import nn
from IPython import display
import numpy as np
from pinn import Pinn
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from torch_cluster import radius_graph, knn_graph # 用于高效邻居搜索
from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import subgraph
import time
import util
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_sparse import SparseTensor
from util import data_process
import math
import torch.nn.functional as F

class ComputeDeDRhoFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xPhys_m, energy_c, coord, pinn):
        ctx.save_for_backward(xPhys_m, coord, energy_c)
        ctx.pinn = pinn
        return energy_c

    @staticmethod
    def backward(ctx, denergy):
        xPhys_m, coord, energy_c = ctx.saved_tensors
        
        with torch.enable_grad():
            gradients = torch.autograd.grad(
                outputs=energy_c,
                inputs=xPhys_m,
                grad_outputs=denergy,
                retain_graph=True,  
                create_graph=True  
            )[0]
        
        return -gradients, None, None, None

class Dmf_tonn(nn.Module):
    def __init__(self, FEA_input, dens_model, disp_model, params, disp_init_epoch, disp_train_epoch,dens_train_epoch,disp_init_lr,disp_train_lr,dens_lr):
        super(Dmf_tonn,self).__init__()
        self.FEA_input = FEA_input
        self.dens_model = dens_model
        self.disp_model = disp_model

        self.keep_shell = params.keep_shell
        self.alpha_init = params.alpha_init
        self.alpha_delta = params.alpha_delta
        self.alpha_max = params.alpha_max        
        self.vf_target = params.vf

        self.pinn = Pinn(self.FEA_input, self.disp_model, params)
        
        # self.disp_init_epoch_now = 0
        self.disp_init_epoch_sum = disp_init_epoch
        # self.disp_train_epoch_now = 0
        self.disp_train_epoch_sum = disp_train_epoch
        self.dens_train_epoch_now = 0
        self.dens_train_epoch_sum = dens_train_epoch


        self.device = torch.device('cuda:0')
        self.disp_init_optimizer = torch.optim.Adam(self.disp_model.parameters(),lr=disp_init_lr)
        self.disp_train_optimizer = torch.optim.Adam(self.disp_model.parameters(),lr=disp_train_lr)
        self.dens_optimizer = torch.optim.Adam(self.dens_model.parameters(),lr=dens_lr)

        self.dlX_dx = self.FEA_input.dlX_dx
        self.dlX_dy = self.FEA_input.dlX_dy
        self.dlX_dz = self.FEA_input.dlX_dz

        self.connect_edge_point_ratio = 1.2


    def fit_disp_init(self):
        for epoch in range(self.disp_init_epoch_sum):
            
            disp_epoch_ratio = epoch / self.disp_init_epoch_sum 

            if epoch%100 ==1:
                print("fit_disp_init ",epoch)
                print(f'loss {loss} epoch {epoch}')

            if self.keep_shell:    
                coord, origin_coord, outside_coor_mask = self.FEA_input.dlX_disp()
                xPhys = torch.where(outside_coor_mask,
                                    torch.ones([coord.shape[0],1], device=self.device),
                                    torch.ones([coord.shape[0],1], device=self.device)*0.5)
            else:
                coord, origin_coord, _ = self.FEA_input.dlX_disp()
                xPhys = torch.ones([coord.shape[0],1], device=self.device)*0.5

            self.disp_init_optimizer.zero_grad()

            loss, _, _  = self.pinn.pinn_init_loss(xPhys, coord, disp_epoch_ratio)
            loss.backward()
            self.disp_init_optimizer.step()

        if self.keep_shell:  
            xPhys = torch.where(self.FEA_input.outside_embedding,
                                torch.ones([self.FEA_input.em_dlX.shape[0],1], device=self.device),
                                torch.ones([self.FEA_input.em_dlX.shape[0],1], device=self.device)*0.5)
        else:
            xPhys = torch.ones([self.FEA_input.em_dlX.shape[0],1], device=self.device)*0.5

        loss, energy_c, u = self.pinn.pinn_init_loss(xPhys, self.FEA_input.em_dlX, disp_epoch_ratio)

        self.c_0 = torch.mean(energy_c).detach()


    def fit_disp_train(self):

        for epoch in range(self.disp_train_epoch_sum):

            disp_epoch_ratio = epoch / self.disp_train_epoch_sum 

            if self.keep_shell:  
                coord, origin_coord, outside_coor_mask = self.FEA_input.dlX_disp()
                xPhys_m = torch.ones((coord.shape[0], 1), device=self.device)
                inside_coord = coord[~outside_coor_mask]
                inside_xPhys_m = self.dens_model(inside_coord, self.dens_epoch_ratio)
                xPhys_m[~outside_coor_mask] = inside_xPhys_m
            else:
                coord, origin_coord, _ = self.FEA_input.dlX_disp()
                xPhys_m = self.dens_model(coord, self.dens_epoch_ratio)

##############
            # if dens_epoch_ratio > 0.6:
            #     xPhys_m, fill_num_ratio = self.dens_filter(xPhys_m, coord, origin_coord)
##############

            self.disp_train_optimizer.zero_grad()

            loss, _, u = self.pinn.pinn_train_loss(xPhys_m, coord, disp_epoch_ratio)
            
            loss.backward()
            self.disp_train_optimizer.step()


###################################################

        # if self.keep_shell:  
        #     xPhys_dlX = torch.ones((self.FEA_input.em_dlX.shape[0], 1), device=self.device)
        #     inside_coord = self.FEA_input.em_dlX[~self.FEA_input.outside_embedding]
        #     inside_xPhys_dlX = self.dens_model(inside_coord, self.dens_epoch_ratio)
        #     xPhys_dlX[~self.FEA_input.outside_embedding] = inside_xPhys_dlX
        # else:
        #     xPhys_dlX = self.dens_model(self.FEA_input.em_dlX, self.dens_epoch_ratio)

##############
        # if dens_epoch_ratio > 0.6:
        #     xPhys_dlX, fill_num_ratio = self.dens_filter(xPhys_dlX, self.FEA_input.em_dlX)
##############            
        # xPhys_dlX = self.dens_model(self.FEA_input.em_dlX, self.dens_epoch_ratio)
        # loss, energy_c, u = self.pinn.pinn_train_loss(xPhys_dlX, self.FEA_input.em_dlX, disp_epoch_ratio)
        # self.c_0 = torch.mean(energy_c)

##############   
        # unreachable_indices, path_graph_mask = self.pinn.disp_filter(u, self.FEA_input.em_dlX)
        # num_in_path = path_graph_mask.sum()
        # return 
##############   

    def fit_dens(self):
        epoch = 0
        connect_point_ratio = 0
        while connect_point_ratio < 0.98 or epoch < self.dens_train_epoch_sum:
            epoch += 1
        # for epoch in range(self.dens_train_epoch_sum):
            self.dens_train_epoch_now += 1
            self.dens_epoch_ratio = self.dens_train_epoch_now / self.dens_train_epoch_sum
            
            self.fit_disp_train()
            
            self.dens_optimizer.zero_grad()

            if self.keep_shell:  
                coord, origin_coord, outside_coor_mask = self.FEA_input.dlX_disp()
                loss, last_epoch_message, connect_point_ratio = self.dens_loss(coord, origin_coord, outside_coor_mask)
            else:
                coord, origin_coord, _ = self.FEA_input.dlX_disp()
                loss, last_epoch_message, connect_point_ratio = self.dens_loss(coord, origin_coord)

            loss.backward()
            self.dens_optimizer.step()
            if epoch > self.dens_train_epoch_sum * 2:
                raise ValueError(f'Connectivity fails to meet requirements')

        return last_epoch_message

  
    def dens_loss(self, coord, origin_coord, outside_coor_mask=None):
 
        conn_teacher_loss_m = 0
        conn_teacher_loss_dlX = 0
        internal_loss_m = 0
        isolated_loss_m = 0
        
        if self.keep_shell:  
            xPhys_m = torch.ones((coord.shape[0], 1), device=self.device)
            inside_coord = coord[~outside_coor_mask]
            inside_xPhys_m = self.dens_model(inside_coord, self.dens_epoch_ratio)
            xPhys_m[~outside_coor_mask] = inside_xPhys_m
        else:
            xPhys_m = self.dens_model(coord, self.dens_epoch_ratio)

        # if self.dens_epoch_ratio > 0.1:    
        #     xPhys_m_clamp, fill_indices, internal_loss_m, isolated_loss_m, connect_point_ratio = self.dens_filter_plan2(xPhys_m, coord)
        #     conn_teacher_loss_m = F.mse_loss(xPhys_m, xPhys_m_clamp)

#################
        # if dens_epoch_ratio > 0.6:
        #     xPhys_m, fill_num_ratio = self.dens_filter(xPhys_m, coord, origin_coord)
#################

        alpha = min(self.alpha_init + self.alpha_delta * self.dens_train_epoch_now , self.alpha_max)

        beta = min(1+0.02* self.dens_train_epoch_now, 10)

        _,energy_c, _ = self.pinn.pinn_init_loss(xPhys_m, coord, 1)
        
        c = torch.mean(ComputeDeDRhoFunction.apply(xPhys_m, energy_c, coord, self.pinn))

        if self.keep_shell:  
            xPhys_dlX = torch.ones((self.FEA_input.em_dlX.shape[0], 1), device=self.device)
            inside_coord = self.FEA_input.em_dlX[~self.FEA_input.outside_embedding]
            inside_xPhys_dlX = self.dens_model(inside_coord, self.dens_epoch_ratio)
            xPhys_dlX[~self.FEA_input.outside_embedding] = inside_xPhys_dlX
        else:    
            xPhys_dlX = self.dens_model(self.FEA_input.em_dlX, self.dens_epoch_ratio)

##############
        # if dens_epoch_ratio > 0.6:
        #     xPhys_dlX, fill_num_ratio = self.dens_filter(xPhys_dlX, self.FEA_input.em_dlX)
        
        # vf = torch.mean(xPhys_dlX)
        # if dens_epoch_ratio > 0.2:
        #     loss = 1.0*c/self.c_0+alpha*(vf/self.vf_target-1.0)**2+alpha*(fill_num_ratio/self.vf_target-1.0)**2 
        # else:
        #     loss = 1.0*c/self.c_0+alpha*(vf/self.vf_target-1.0)**2 
##############     

        internal_loss_dlX = 0
        isolated_loss_dlX = 0
        connect_point_ratio = 0
        disp_internal_loss = 0
        disp_isolated_loss = 0

        if self.keep_shell:  
            vf = torch.mean(xPhys_dlX) - self.FEA_input.outside_ele_ratio
        else:
            vf = torch.mean(xPhys_dlX)

        if self.dens_epoch_ratio > 0.05:
        # if abs(vf - self.vf_target) / self.vf_target < 0.5:
            _, _, u_dlX = self.pinn.pinn_init_loss(xPhys_dlX, self.FEA_input.em_dlX, 1)
            xPhys_dlX_clamp, fill_indices, internal_loss_dlX, isolated_loss_dlX, connect_point_ratio = self.dens_filter_plan2(xPhys_dlX, vf, self.FEA_input.em_dlX)
            conn_teacher_loss_dlX = F.mse_loss(xPhys_dlX, xPhys_dlX_clamp)
            # disp_internal_loss, disp_isolated_loss = self.disp_connect_filter(xPhys_dlX, u_dlX, fill_indices, self.FEA_input.em_dlX)
            

        
        loss_c = 1.0*c/self.c_0 
        loss_vf = alpha*(vf/self.vf_target-1.0)**2 
        loss_dens = internal_loss_dlX + isolated_loss_dlX
        # loss_dens = isolated_loss_dlX
        # loss_dens = 0 
        # loss_disp =  disp_internal_loss * alpha / 10 + disp_isolated_loss * alpha
        loss_disp = 0

        loss = loss_c + loss_vf + loss_dens + loss_disp
        # loss = 1.0*c/self.c_0+alpha*(vf/self.vf_target-1.0)**2

        if self.dens_train_epoch_now % 3 == 0:
            print(f'Epoch: {self.dens_train_epoch_now}')
            print(f'Compliance: {c.item()}')
            print(f'VF: {vf.item()}')
            print(f'connect_point_ratio: {connect_point_ratio}')
            print(f'energy loss {c/self.c_0}, vf loss {(vf/self.vf_target-1.0)**2}, internal loss {internal_loss_dlX}, isolated loss {isolated_loss_dlX}')
            # print(f'Total Loss: {loss.item()}')

        last_epoch_message = []
        if self.dens_train_epoch_now == self.dens_train_epoch_sum:
            last_epoch_message = [
                f'Epoch: {self.dens_train_epoch_now}', 
                f'Compliance: {c.item()}', 
                f'VF: {vf.item()}', 
                f'connect_point_ratio: {connect_point_ratio}', 
                f'energy loss {c/self.c_0}, vf loss {(vf/self.vf_target-1.0)**2}, internal loss {internal_loss_dlX}, isolated loss {isolated_loss_dlX}'
            ]


        return loss, last_epoch_message, connect_point_ratio

###################################################     
    def dens_filter(self, xPhys_m, coord, origin_coord=None):

        target_point_num = int(coord.shape[0] * self.vf_target)
        next_voxel_num = 1

        connect_test_point_num = int(target_point_num * self.connect_edge_point_ratio)
        xPhys_m_sorted, xPhys_m_indices = torch.sort(xPhys_m, dim=0, descending=True, stable=True)
        if origin_coord is None:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.02 * next_voxel_num
        else:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.5 * next_voxel_num

        # print(f'coord num {coord_num}')
        connect_test_point_indices = xPhys_m_indices[:connect_test_point_num].squeeze()
        connect_test_point_coord = coord[connect_test_point_indices]

        edge_index = radius_graph(connect_test_point_coord, r=r, loop=False, max_num_neighbors=64,)
        # print(f'edge num {edge_index.shape[1]}')  
        row, col = edge_index
        coord_i = connect_test_point_coord[row]  
        coord_j = connect_test_point_coord[col] 
        diffs = torch.abs(coord_i - coord_j)

        if origin_coord is None:
            thresholds = torch.tensor([self.dlX_dy*1.02*next_voxel_num, self.dlX_dx*1.02*next_voxel_num, self.dlX_dz*1.02*next_voxel_num], device=self.device)
            valid_edge_mask = (diffs <= thresholds).all(dim=1)
        else:
            origin_thresholds = torch.tensor([self.dlX_dy*1.02*next_voxel_num, self.dlX_dx*1.02*next_voxel_num, self.dlX_dz*1.02*next_voxel_num], device=self.device)
            origin_connect_test_point_coord = origin_coord[connect_test_point_indices]
            origin_coord_i = origin_connect_test_point_coord[row]  
            origin_coord_j = origin_connect_test_point_coord[col] 
            origin_diffs = torch.abs(origin_coord_i - origin_coord_j)
            thresholds = torch.tensor([self.dlX_dy*1.5*next_voxel_num, self.dlX_dx*1.5*next_voxel_num, self.dlX_dz*1.5*next_voxel_num], device=self.device)
            valid_edge_mask = ((origin_diffs <= origin_thresholds) & (diffs <= thresholds)).all(dim=1)

        # print(f'valid edge num {valid_edge_mask.sum()}')
        edge_index_valid = edge_index[:,valid_edge_mask]
        connect_test_point_array = torch.arange(connect_test_point_num,device=self.device).view(-1, 1)
        data = Data(x=connect_test_point_array, edge_index=edge_index_valid, num_nodes=connect_test_point_num)

        transform = LargestConnectedComponents(num_components=1,connection='weak')
        transformed_data = transform(data)
        node_indices = transformed_data.x.squeeze().long()

        self.connect_edge_point_ratio -= (node_indices.shape[0] - target_point_num) / target_point_num * 0.9
        if node_indices.shape[0] >= target_point_num:
            in_connect_mask = torch.zeros(connect_test_point_indices.shape[0], dtype=torch.bool, device=self.device)
            in_connect_mask[node_indices] = True
            fill_indices = connect_test_point_indices[in_connect_mask][:target_point_num]
        else:
            alter_indices = connect_test_point_num - node_indices.shape[0]
            xPhys_indices = xPhys_m_indices.squeeze()
            fill_indices = torch.cat([connect_test_point_indices[node_indices], xPhys_indices[connect_test_point_num:connect_test_point_num + alter_indices]], dim=0)
            
        
        fill_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        fill_indices_mask[fill_indices] = True

        # end_time = time.perf_counter()  
        # elapsed_time = end_time - start_time 
        # print(f"get connect compoment used time : {elapsed_time:.6f} s")

        internal_points = xPhys_m[fill_indices_mask]
        internal_loss = torch.mean(1 - internal_points) 

        isolated_points = xPhys_m[~fill_indices_mask]
        isolated_loss = torch.mean(isolated_points)

        xPhys_m_clamp = torch.where(fill_indices_mask, torch.ones_like(xPhys_m), torch.zeros_like(xPhys_m))        
        connect_point_ratio = len(node_indices) /  connect_test_point_num

        #########################
        # if self.dens_train_epoch_now % 60 == 0: 
            # self.fill_indices = fill_indices
            # self.disp_connect_filter(u, coord, fill_indices, transformed_data)
            # self.disp_connect_filter(u, coord, fill_indices, origin_coord)

        #########################

        return xPhys_m_clamp, fill_indices, internal_loss, isolated_loss, connect_point_ratio

    def dens_filter_plan2(self, xPhys_m, vf, coord, origin_coord=None):
        # connect_test_ratio = max(self.vf_target, min(1.0, self.vf_target * (1.1 - dens_epoch_ratio)))

        # start_time = time.perf_counter()  

        next_voxel_num = 1

        if self.keep_shell:  
            connect_test_ratio = max(self.FEA_input.outside_ele_ratio + self.vf_target, 
                                    (self.FEA_input.outside_ele_ratio + self.vf_target) * (1.3 - 0.5 * self.dens_epoch_ratio))
        else:
            connect_test_ratio = max(self.vf_target, self.vf_target * (1.5 - 1.0 * self.dens_epoch_ratio))
            # connect_test_ratio = max(self.vf_target, self.vf_target * (3.0 - 2.0 * self.dens_epoch_ratio))
            # connect_test_ratio = max(self.vf_target, min(1.0, vf * (1.5 - 1.0 * self.dens_epoch_ratio)))
        # connect_test_ratio = self.vf_target
        connect_test_point_num = int(coord.shape[0] * connect_test_ratio)
        xPhys_m_sorted, xPhys_m_indices = torch.sort(xPhys_m, dim=0, descending=True, stable=True)
        if origin_coord is None:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.02 * next_voxel_num
        else:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.5 * next_voxel_num

        # print(f'coord num {coord_num}')
        connect_test_point_indices = xPhys_m_indices[:connect_test_point_num].squeeze()

        connect_test_point_coord = coord[connect_test_point_indices]
        edge_index = radius_graph(connect_test_point_coord, r=r, loop=False, max_num_neighbors=64,)
        # print(f'edge num {edge_index.shape[1]}')  

        row, col = edge_index
        coord_i = connect_test_point_coord[row]  
        coord_j = connect_test_point_coord[col] 
        diffs = torch.abs(coord_i - coord_j)

        if origin_coord is None:
            thresholds = torch.tensor([self.dlX_dy*1.02*next_voxel_num, self.dlX_dx*1.02*next_voxel_num, self.dlX_dz*1.02*next_voxel_num], device=self.device)
            valid_edge_mask = (diffs <= thresholds).all(dim=1)
        else:
            origin_thresholds = torch.tensor([self.dlX_dy*1.02*next_voxel_num, self.dlX_dx*1.02*next_voxel_num, self.dlX_dz*1.02*next_voxel_num], device=self.device)
            origin_connect_test_point_coord = origin_coord[connect_test_point_indices]
            origin_coord_i = origin_connect_test_point_coord[row]  
            origin_coord_j = origin_connect_test_point_coord[col] 
            origin_diffs = torch.abs(origin_coord_i - origin_coord_j)
            thresholds = torch.tensor([self.dlX_dy*1.5*next_voxel_num, self.dlX_dx*1.5*next_voxel_num, self.dlX_dz*1.5*next_voxel_num], device=self.device)
            valid_edge_mask = ((origin_diffs <= origin_thresholds) & (diffs <= thresholds)).all(dim=1)

        # print(f'valid edge num {valid_edge_mask.sum()}')
        edge_index_valid = edge_index[:,valid_edge_mask]
        connect_test_point_array = torch.arange(connect_test_point_num,device=self.device).view(-1, 1)
        data = Data(x=connect_test_point_array, edge_index=edge_index_valid, num_nodes=connect_test_point_num)

        transform = LargestConnectedComponents(num_components=1,connection='weak')
        transformed_data = transform(data)
        node_indices = transformed_data.x.squeeze().long()
        
        fill_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        fill_indices = connect_test_point_indices[node_indices]
        fill_indices_mask[fill_indices] = True

        # end_time = time.perf_counter()  
        # elapsed_time = end_time - start_time 
        # print(f"get connect compoment used time : {elapsed_time:.6f} s")

        unfill_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        unfill_indices = torch.where(~torch.isin(connect_test_point_array, node_indices))[0]
        unfill_indices_mask[unfill_indices] = True

        alter_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        alter_indices = xPhys_m_indices[connect_test_point_num:connect_test_point_num+len(unfill_indices)]
        alter_indices_mask[alter_indices] = True

        internal_points = xPhys_m[fill_indices_mask]
        internal_loss = torch.mean(1 - internal_points) 

        
        # isolated_points_index = torch.where(~fill_indices_mask & (xPhys_m > 0.001))[0]
        # isolated_points = xPhys_m[isolated_points_index]
        isolated_points = xPhys_m[~fill_indices_mask]
        isolated_loss = torch.mean(isolated_points)

        xPhys_m_clamp = torch.where(fill_indices_mask, torch.ones_like(xPhys_m), torch.zeros_like(xPhys_m))        
        connect_point_ratio = len(node_indices) /  connect_test_point_num

        #########################
        # if self.dens_train_epoch_now % 60 == 0: 
            # self.fill_indices = fill_indices
            # self.disp_connect_filter(u, coord, fill_indices, transformed_data)
            # self.disp_connect_filter(u, coord, fill_indices, origin_coord)

        #########################

        return xPhys_m_clamp, fill_indices, internal_loss, isolated_loss, connect_point_ratio
    

    # def disp_connect_filter(self, u, coord, fill_indices, transformed_data):
    def disp_connect_filter(self, xPhys_dlX, u, fill_indices, coord, origin_coord=None):

        next_voxel_num = 1.42

        fill_u_each_point = torch.norm(u[fill_indices], dim=1, keepdim=False)
        fill_u_dir = u[fill_indices] / (torch.norm(u[fill_indices], dim=1, keepdim=True) + 1e-8)

        fill_coord = coord[fill_indices]
        # fill_edge_i, fill_edge_j = transformed_data.edge_index
        # ''' 
        if origin_coord is None:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.02 * next_voxel_num
        else:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.5 * next_voxel_num

        edge_index = radius_graph(fill_coord, r=r, loop=False, max_num_neighbors=64,)
        row, col = edge_index
        coord_i = fill_coord[row]  
        coord_j = fill_coord[col] 
        diffs = torch.abs(coord_i - coord_j)
        if origin_coord is None:
            thresholds = torch.tensor([self.dlX_dy*1.02*next_voxel_num, self.dlX_dx*1.02*next_voxel_num, self.dlX_dz*1.02*next_voxel_num], device=self.device)
            valid_edge_mask = (diffs <= thresholds).all(dim=1)
        else:
            origin_thresholds = torch.tensor([self.dlX_dy*1.02*next_voxel_num, self.dlX_dx*1.02*next_voxel_num, self.dlX_dz*1.02*next_voxel_num], device=self.device)
            origin_connect_test_point_coord = origin_coord[fill_indices]
            origin_coord_i = origin_connect_test_point_coord[row]  
            origin_coord_j = origin_connect_test_point_coord[col] 
            origin_diffs = torch.abs(origin_coord_i - origin_coord_j)
            thresholds = torch.tensor([self.dlX_dy*1.5*next_voxel_num, self.dlX_dx*1.5*next_voxel_num, self.dlX_dz*1.5*next_voxel_num], device=self.device)
            valid_edge_mask = ((origin_diffs <= origin_thresholds) & (diffs <= thresholds)).all(dim=1)
        edge_index_valid = edge_index[:,valid_edge_mask]
        fill_edge_i, fill_edge_j = edge_index_valid
        # '''

        fill_u_edge_i = fill_u_each_point[fill_edge_i]
        fill_u_edge_j = fill_u_each_point[fill_edge_j]

        # dir_mask = (fill_u_edge_i >= fill_u_edge_j).flatten()
        # dir_mask = ((fill_u_edge_i >= fill_u_edge_j) | ((fill_u_edge_i - fill_u_edge_j).abs() < 1e-4)).flatten()

        fill_u_dir_i = fill_u_dir[fill_edge_i]
        fill_u_dir_j = fill_u_dir[fill_edge_j]

        dir_mask = ((fill_u_edge_i >= fill_u_edge_j) & ((fill_u_dir_i * fill_u_dir_j).sum(dim=-1) > math.cos(30))).flatten()

        fill_u_each_point_epsilon = fill_u_each_point.max() * 0.001
        equ_mask = (((fill_u_edge_i - fill_u_edge_j).abs() < fill_u_each_point_epsilon) & ((fill_u_dir_i * fill_u_dir_j).sum(dim=-1) > math.cos(30))).flatten()

        num_dir = dir_mask.sum()
        num_equ = equ_mask.sum()

        dir_edges = torch.stack([
            torch.where(dir_mask, fill_edge_i, fill_edge_j),
            torch.where(dir_mask, fill_edge_j, fill_edge_i)], dim=0)

        equ_edges = torch.cat([
            torch.stack([fill_edge_i[equ_mask], fill_edge_j[equ_mask]], dim=0),
            torch.stack([fill_edge_j[equ_mask], fill_edge_i[equ_mask]], dim=0)], dim=1)
        
        directed_edges = torch.cat([dir_edges, equ_edges], dim=1)

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

        if self.dens_train_epoch_now == self.dens_train_epoch_sum:
            unreachable_indices 
            path_graph_mask 

        

        # '''
        if self.dens_train_epoch_now % 120 == 0: 
            data_show = data_process()
            fixed_indices_global = self.FEA_input.fixed_voxel_index
            force_indices_global = self.FEA_input.force_voxel_index
            # xyz
            xPhys_reachable_full = np.zeros((self.FEA_input.voxel_Ny, self.FEA_input.voxel_Nx, self.FEA_input.voxel_Nz))
            xPhys_reachable = np.zeros(coord.shape[0])
            xPhys_reachable[fill_indices.detach().cpu().numpy()] = 1
            xPhys_reachable_full[np.transpose(self.FEA_input.voxelgrid, (1,0,2))] = xPhys_reachable
            
            unreachable_indices_fill = fill_indices[unreachable_indices].detach().cpu().numpy()
            unreachable_indices_global_flat = np.flatnonzero(np.transpose(self.FEA_input.voxelgrid, (1,0,2)))[unreachable_indices_fill]
            unreachable_indices_yxz = np.array(np.unravel_index(unreachable_indices_global_flat, (self.FEA_input.voxel_Ny, self.FEA_input.voxel_Nx, self.FEA_input.voxel_Nz))).T
            unreachable_indices_global = unreachable_indices_yxz[:, [1,0,2]]
            
            reachable_indices_fill = fill_indices[path_graph_mask].detach().cpu().numpy()
            reachable_indices_global_flat = np.flatnonzero(np.transpose(self.FEA_input.voxelgrid, (1,0,2)))[reachable_indices_fill]
            reachable_indices_yxz = np.array(np.unravel_index(reachable_indices_global_flat, (self.FEA_input.voxel_Ny, self.FEA_input.voxel_Nx, self.FEA_input.voxel_Nz))).T
            reachable_indices_global = reachable_indices_yxz[:, [1,0,2]]
        
            data_show.show_u_reachable(xPhys_reachable_full, coord, fixed_indices_global, force_indices_global, unreachable_indices_global, reachable_indices_global)
        # '''

        reachable_indices_fill = fill_indices[path_graph_mask]
        unreachable_indices_fill = fill_indices[unreachable_indices]

        disp_internal_points = xPhys_dlX[reachable_indices_fill]
        disp_internal_loss = torch.mean(1 - disp_internal_points) 

        disp_isolated_points = xPhys_dlX[unreachable_indices_fill]
        disp_isolated_loss = torch.mean(disp_isolated_points)
        return disp_internal_loss, disp_isolated_loss

        
    def bfs_sparse_tensor_multi_source(self, edge_index, start_index, num_index, max_steps=None, batch_frontier=False):

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

    def dens_filter_energy(self, energy_each_point, xPhys_m, coord, origin_coord=None):

        # start_time = time.perf_counter()  

        dens_epoch_ratio = self.dens_train_epoch_now / self.dens_train_epoch_sum  

        if self.keep_shell:  
            connect_test_ratio = max(self.FEA_input.outside_ele_ratio + self.vf_target, 
                                    (self.FEA_input.outside_ele_ratio + self.vf_target) * (1.3 - 0.5 * dens_epoch_ratio))
        else:
            connect_test_ratio = max(self.vf_target, self.vf_target * (1.3 - 0.5 * dens_epoch_ratio))
        # connect_test_ratio = self.vf_target
        connect_test_point_num = int(coord.shape[0] * connect_test_ratio)
        xPhys_m_sorted, xPhys_m_indices = torch.sort(xPhys_m, dim=0, descending=True, stable=True)
        if origin_coord is None:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.02
        else:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.5

        # print(f'coord num {coord_num}')
        connect_test_point_indices = xPhys_m_indices[:connect_test_point_num].squeeze()

        connect_test_point_coord = coord[connect_test_point_indices]
        edge_index = radius_graph(connect_test_point_coord, r=r, loop=False, max_num_neighbors=64,)
        # print(f'edge num {edge_index.shape[1]}')  

        row, col = edge_index
        coord_i = connect_test_point_coord[row]  
        coord_j = connect_test_point_coord[col] 
        diffs = torch.abs(coord_i - coord_j)

        if origin_coord is None:
            thresholds = torch.tensor([self.dlX_dy*1.02, self.dlX_dx*1.02, self.dlX_dz*1.02], device=self.device)
            valid_edge_mask = (diffs <= thresholds).all(dim=1)
        else:
            origin_thresholds = torch.tensor([self.dlX_dy*1.02, self.dlX_dx*1.02, self.dlX_dz*1.02], device=self.device)
            origin_connect_test_point_coord = origin_coord[connect_test_point_indices]
            origin_coord_i = origin_connect_test_point_coord[row]  
            origin_coord_j = origin_connect_test_point_coord[col] 
            origin_diffs = torch.abs(origin_coord_i - origin_coord_j)
            thresholds = torch.tensor([self.dlX_dy*1.5, self.dlX_dx*1.5, self.dlX_dz*1.5], device=self.device)
            valid_edge_mask = ((origin_diffs <= origin_thresholds) & (diffs <= thresholds)).all(dim=1)

        # print(f'valid edge num {valid_edge_mask.sum()}')
        edge_index_valid = edge_index[:,valid_edge_mask]
        connect_test_point_array = torch.arange(connect_test_point_num,device=self.device).view(-1, 1)
        data = Data(x=connect_test_point_array, edge_index=edge_index_valid, num_nodes=connect_test_point_num)

        transform = LargestConnectedComponents(num_components=1,connection='weak')
        transformed_data = transform(data)
        node_indices = transformed_data.x.squeeze().long()
        
        fill_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        fill_indices = connect_test_point_indices[node_indices]
        fill_indices_mask[fill_indices] = True

        # end_time = time.perf_counter()  
        # elapsed_time = end_time - start_time 
        # print(f"get connect compoment used time : {elapsed_time:.6f} s")

        unfill_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        unfill_indices = torch.where(~torch.isin(connect_test_point_array, node_indices))[0]
        unfill_indices_mask[unfill_indices] = True

        alter_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        alter_indices = xPhys_m_indices[connect_test_point_num:connect_test_point_num+len(unfill_indices)]
        alter_indices_mask[alter_indices] = True


        internal_points = xPhys_m[fill_indices_mask]
        internal_loss = torch.mean(1 - internal_points) 

        isolated_points = xPhys_m[~fill_indices_mask]
        isolated_loss = torch.mean(isolated_points)

        xPhys_m_clamp = torch.where(fill_indices_mask, torch.ones_like(xPhys_m), torch.zeros_like(xPhys_m))        
        connect_point_ratio = len(node_indices) /  connect_test_point_num


        fill_energy_each_point = energy_each_point[fill_indices]
        fill_coord = coord[fill_indices]
        fill_edge_i, fill_edge_j = transformed_data.edge_index 
        fill_energy_edge_i = fill_energy_each_point[fill_edge_i]
        fill_energy_edge_j = fill_energy_each_point[fill_edge_j]
        dir_mask = (fill_energy_edge_i > fill_energy_edge_j).flatten()
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

        if self.dens_train_epoch_now == self.dens_train_epoch_sum:
            unreachable_indices
            path_graph_mask  


        fixed_indices_global = self.FEA_input.fixed_voxel_index
        force_indices_global = self.FEA_input.force_voxel_index
        data_show = data_process()
        xPhys_m_full = np.zeros((self.FEA_input.voxel_Ny, self.FEA_input.voxel_Nx, self.FEA_input.voxel_Nz))
        xPhys_m_full[np.transpose(self.FEA_input.voxelgrid, (1,0,2))] = xPhys_m.view(-1).cpu().detach().numpy()
        data_show.show_energy_distribution(energy_each_point, xPhys_m_full, coord, fixed_indices_global, force_indices_global)

        return xPhys_m_clamp, internal_loss, isolated_loss, connect_point_ratio

    def dens_filter_backup(self, xPhys_m, coord, origin_coord=None):

        # start_time = time.perf_counter()  

        dens_epoch_ratio = self.dens_train_epoch_now / self.dens_train_epoch_sum  

        if self.keep_shell:  
            connect_test_ratio = max(self.FEA_input.outside_ele_ratio + self.vf_target, 
                                    (self.FEA_input.outside_ele_ratio + self.vf_target) * (1.3 - 0.5 * dens_epoch_ratio))
        else:
            connect_test_ratio = max(self.vf_target, self.vf_target * (1.3 - 0.5 * dens_epoch_ratio))
        connect_test_point_num = int(coord.shape[0] * connect_test_ratio)
        xPhys_m_sorted, xPhys_m_indices = torch.sort(xPhys_m, dim=0, descending=True, stable=True)
        if origin_coord is None:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.02
        else:
            r = max(self.dlX_dx, self.dlX_dy, self.dlX_dz) * 1.5

        # print(f'coord num {coord_num}')
        connect_test_point_indices = xPhys_m_indices[:connect_test_point_num].squeeze()

        connect_test_point_coord = coord[connect_test_point_indices]
        edge_index = radius_graph(connect_test_point_coord, r=r, loop=False, max_num_neighbors=64,)
        # print(f'edge num {edge_index.shape[1]}')  

        row, col = edge_index
        coord_i = connect_test_point_coord[row]  
        coord_j = connect_test_point_coord[col] 
        diffs = torch.abs(coord_i - coord_j)

        if origin_coord is None:
            thresholds = torch.tensor([self.dlX_dy*1.02, self.dlX_dx*1.02, self.dlX_dz*1.02], device=self.device)
            valid_edge_mask = (diffs <= thresholds).all(dim=1)
        else:
            origin_thresholds = torch.tensor([self.dlX_dy*1.02, self.dlX_dx*1.02, self.dlX_dz*1.02], device=self.device)
            origin_connect_test_point_coord = origin_coord[connect_test_point_indices]
            origin_coord_i = origin_connect_test_point_coord[row]  
            origin_coord_j = origin_connect_test_point_coord[col] 
            origin_diffs = torch.abs(origin_coord_i - origin_coord_j)
            thresholds = torch.tensor([self.dlX_dy*1.5, self.dlX_dx*1.5, self.dlX_dz*1.5], device=self.device)
            valid_edge_mask = ((origin_diffs <= origin_thresholds) & (diffs <= thresholds)).all(dim=1)

        # print(f'valid edge num {valid_edge_mask.sum()}')
        edge_index_valid = edge_index[:,valid_edge_mask]
        connect_test_point_array = torch.arange(connect_test_point_num,device=self.device).view(-1, 1)
        data = Data(x=connect_test_point_array, edge_index=edge_index_valid, num_nodes=connect_test_point_num)

        transform = LargestConnectedComponents(num_components=1,connection='weak')

        transformed_data = transform(data)
        node_indices = transformed_data.x.squeeze().long()
        
        fill_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        fill_indices = connect_test_point_indices[node_indices]

        fill_indices_mask[fill_indices] = True

        # end_time = time.perf_counter()  
        # elapsed_time = end_time - start_time 
        # print(f"get connect compoment used time : {elapsed_time:.6f} s")

        unfill_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        unfill_indices = torch.where(~torch.isin(connect_test_point_array, node_indices))[0]

        unfill_indices_mask[unfill_indices] = True


        alter_indices_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        alter_indices = xPhys_m_indices[connect_test_point_num:connect_test_point_num+len(unfill_indices)]
        alter_indices_mask[alter_indices] = True


        internal_points = xPhys_m[fill_indices_mask]
        internal_loss = torch.mean(1 - internal_points) 

        isolated_points = xPhys_m[~fill_indices_mask]

        isolated_loss = torch.mean(isolated_points)

        xPhys_m_clamp = torch.where(fill_indices_mask, torch.ones_like(xPhys_m), torch.zeros_like(xPhys_m))        

        connect_point_ratio = len(node_indices) /  connect_test_point_num

        return xPhys_m_clamp, internal_loss, isolated_loss, connect_point_ratio

    def energy_flow_mask(self, xPhys_m, coord, energy_grad, fixed_mask, r):
        edge_index = radius_graph(coord, r=r, loop=False, max_num_neighbors=32)
        row, col = edge_index
        dir_vec = coord[col] - coord[row]
        dir_vec = dir_vec / (torch.norm(dir_vec, dim=1, keepdim=True) + 1e-8)

        grad_vec = energy_grad[row]
        grad_vec = grad_vec / (torch.norm(grad_vec, dim=1, keepdim=True) + 1e-8)

        cos_sim = (dir_vec * grad_vec).sum(dim=1)
        valid_flow_mask = (cos_sim > 0.5) 
        energy_flow_edges = edge_index[:, valid_flow_mask]

        data = Data(x=torch.arange(coord.shape[0], device=self.device), edge_index=energy_flow_edges)
        transform = LargestConnectedComponents(num_components=1, connection='weak')
        transformed_data = transform(data)
        node_indices = transformed_data.x.squeeze().long()

        energy_flow_mask = torch.zeros_like(xPhys_m, dtype=torch.bool)
        energy_flow_mask[node_indices] = True
        return energy_flow_mask