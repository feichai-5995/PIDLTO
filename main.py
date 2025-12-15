import os
import numpy as np
import time
import argparse
from omegaconf import OmegaConf
import torch
from datetime import datetime

from read_stl import Read_stl
from voxel_writer import Voxel_writer
from stl_to_voxel import Stl_to_voxel
from voxel_to_FEA_input import Voxel_to_FEA_input
from basic_net import Dens_Net, Disp_Net
from dmf_tonn import Dmf_tonn
from util import data_process



class main_process:
    def __init__(self):
        parser = argparse.ArgumentParser(prog='myprogram')
        parser.add_argument('--config', default='./config.yaml', help='yaml file path')

        self.args = parser.parse_args() 
        self.config = OmegaConf.load(self.args.config)
        OmegaConf.resolve(self.config)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 

        self.util = data_process()

        self.disp_init_epoch = self.config.epoch_and_lr.disp_init_epoch
        self.disp_train_epoch = self.config.epoch_and_lr.disp_train_epoch
        self.dens_train_epoch = self.config.epoch_and_lr.dens_train_epoch
        self.disp_init_lr = self.config.epoch_and_lr.disp_init_lr
        self.disp_train_lr = self.config.epoch_and_lr.disp_train_lr
        self.dens_lr = self.config.epoch_and_lr.dens_lr

    def get_stl(self):
        full_read_file_name = f'./data/{self.config.file.model_name}.stl'
        stl = Read_stl(full_read_file_name)
        self.meshVertexs = stl()[0]

        self.mvoxel = Stl_to_voxel(self.config.stl_to_voxel, predict_label=False)
        self.mvoxel.gen_vox_grid(self.meshVertexs,self.config.stl_to_voxel.ray,self.config.stl_to_voxel.parallel)
        self.mvoxel.gen_vox_info()
        full_save_png_name = f"./save/origin_model_png/origin_model_{self.config.file.model_name}_{self.timestamp}.png"
        self.mvoxel.showVoxel(full_save_png_name)

        # mesh = Voxel_writer(self.config.voxel_writer, self.mvoxel.nod_coor_abs, self.mvoxel.ele_nod)
        # mesh()
    
    def train(self):
        self.FEA_input = Voxel_to_FEA_input(self.config.FEA, self.mvoxel, self.meshVertexs, predict_label=False)
        full_save_png_name = f"./save/input_model_png/input_model_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_input_fix_force(self.mvoxel.voxelgrid, self.FEA_input.fixed_voxel_index, self.FEA_input.force_voxel_index, full_save_png_name)
        self.dens_model = Dens_Net().to('cuda:0')
        self.disp_model = Disp_Net().to('cuda:0')
        self.opt = Dmf_tonn(self.FEA_input, self.dens_model, self.disp_model, self.config.FEA, 
                            self.disp_init_epoch, self.disp_train_epoch, self.dens_train_epoch, self.disp_init_lr, self.disp_train_lr, self.dens_lr)
        self.opt.fit_disp_init()
        self.last_fit_message = self.opt.fit_dens()

    def test_same_ratio(self):
        xPhys_dlX = self.dens_model(self.FEA_input.em_dlX, 1.0)
        self.xPhys_dlX_full = np.zeros((self.FEA_input.voxel_Ny, self.FEA_input.voxel_Nx, self.FEA_input.voxel_Nz))
        self.xPhys_dlX_full[np.transpose(self.FEA_input.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
        full_save_png_name = f"./save/output_model_png/output_model_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_output_fix_force(self.xPhys_dlX_full, self.FEA_input.fixed_voxel_index, self.FEA_input.force_voxel_index, full_save_png_name)
        full_save_png_name = f"./save/output_model_png/output_mesh_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_mesh(self.xPhys_dlX_full, self.FEA_input.fixed_voxel_index, self.FEA_input.force_voxel_index, full_save_png_name, self.config.FEA.vf, dens_limits=0.5)

        xPhys_dlX, fill_indices, _, _, _ = self.opt.dens_filter_plan2(xPhys_dlX, self.config.FEA.vf, self.FEA_input.em_dlX)
        energy_c = self.opt.pinn.pinn_predict_loss(xPhys_dlX, self.FEA_input.em_dlX, 1)
        self.c = torch.mean(energy_c)
        self.vf = torch.mean(xPhys_dlX) 
        print(f'finial VF: {self.vf.item()}')
        print(f'finial Compliance: {self.c.item()}')
        self.xPhys_dlX_full = np.zeros((self.FEA_input.voxel_Ny, self.FEA_input.voxel_Nx, self.FEA_input.voxel_Nz))
        self.xPhys_dlX_full[np.transpose(self.FEA_input.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
        full_save_png_name = f"./save/output_model_png/output_model_filtered_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_output_fix_force(self.xPhys_dlX_full, self.FEA_input.fixed_voxel_index, self.FEA_input.force_voxel_index, full_save_png_name)
        full_save_png_name = f"./save/output_model_png/output_mesh_filtered_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_mesh(self.xPhys_dlX_full, self.FEA_input.fixed_voxel_index, self.FEA_input.force_voxel_index, full_save_png_name, self.config.FEA.vf, dens_limits=0.5)
    
    def test_multi_ratio(self):
        mvoxel_predict = Stl_to_voxel(self.config.stl_to_voxel, predict_label=True)
        mvoxel_predict.gen_vox_grid(self.meshVertexs,self.config.stl_to_voxel.ray,self.config.stl_to_voxel.parallel)
        mvoxel_predict.gen_vox_info()
        full_save_png_name = f"./save/origin_model_png/origin_pridict_model_{self.config.file.model_name}_{self.timestamp}.png"
        # self.mvoxel_predict.showVoxel(full_save_png_name)
        FEA_input_predict = Voxel_to_FEA_input(self.config.FEA, mvoxel_predict, self.meshVertexs, predict_label=True)
        full_save_png_name = f"./save/input_model_png/input_pridict_model_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_input_fix_force(mvoxel_predict.voxelgrid, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name)
        
        xPhys_dlX = self.dens_model(FEA_input_predict.em_dlX, 1.0)
        self.xPhys_dlX_full = np.zeros((FEA_input_predict.voxel_Ny, FEA_input_predict.voxel_Nx, FEA_input_predict.voxel_Nz))
        self.xPhys_dlX_full[np.transpose(FEA_input_predict.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
        full_save_png_name = f"./save/output_model_png/output_pridict_model_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_output_fix_force(self.xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name)
        full_save_png_name = f"./save/output_model_png/output_pridict_mesh_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_mesh( self.xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name, self.config.FEA.vf, dens_limits=0.5)

        xPhys_dlX, fill_indices, _, _, _ = self.opt.dens_filter(xPhys_dlX, FEA_input_predict.em_dlX)
        energy_c = self.opt.pinn.pinn_predict_loss(xPhys_dlX, FEA_input_predict.em_dlX, 1)
        self.c = torch.mean(energy_c)
        self.vf = torch.mean(xPhys_dlX)
        print(f'finial VF: {self.vf.item()}')
        print(f'finial Compliance: {self.c.item()}')
        self.xPhys_dlX_full = np.zeros((FEA_input_predict.voxel_Ny, FEA_input_predict.voxel_Nx, FEA_input_predict.voxel_Nz))
        self.xPhys_dlX_full[np.transpose(FEA_input_predict.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
        full_save_png_name = f"./save/output_model_png/output_pridict_model_filtered_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_output_fix_force(self.xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name)
        full_save_png_name = f"./save/output_model_png/output_pridict_mesh_filtered_{self.config.file.model_name}_{self.timestamp}.png"
        self.util.show_mesh(self.xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name, self.config.FEA.vf, dens_limits=0.5)

    def save_message(self):
        full_save_file_name = f"./save/stl/topo_{self.config.file.model_name}_{self.timestamp}.stl"
        mesh = self.util.save_stl(self.xPhys_dlX_full, full_save_file_name, self.config.FEA.vf, dens_limits=0.5)

        full_save_txt_name = f"./save/txt/message_{self.config.file.model_name}.txt"
        message = [
            f'timestamp: {self.timestamp}', 
            f'finial VF: {self.vf.item()}',
            f'finial Compliance: {self.c.item()}',
            *self.last_fit_message,

            f'voxel_z_size: {self.config.stl_to_voxel.voxel_z_size}',
            f'load_posi_ratio: {self.config.FEA.load_posi_ratio}',
            f'load_dire: {self.config.FEA.load_dire}',
            f'fixed_voxel_start: {self.config.FEA.fixed_voxel_start}',
            f'fixed_voxel_orientation: {self.config.FEA.fixed_voxel_orientation}',
            f'fixed_voxel_num: {self.config.FEA.fixed_voxel_num}',
            f'target vf: {self.config.FEA.vf}',
            
            f'disp_init_epoch: {self.disp_init_epoch}',
            f'disp_train_epoch: {self.disp_train_epoch}',
            f'dens_train_epoch: {self.dens_train_epoch}',
            f'disp_init_lr: {self.disp_init_lr}',
            f'disp_train_lr: {self.disp_train_lr}',
            f'dens_lr: {self.dens_lr}'
        ]
        self.util.save_message_to_txt(full_save_txt_name, message)

"""
def main():
    parser = argparse.ArgumentParser(prog='myprogram')
    parser.add_argument('--config', default='./config.yaml', help='yaml file path')

    args = parser.parse_args() 
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    full_read_file_name = f'./data/{config.file.model_name}.stl'
    stl = Read_stl(full_read_file_name)
    meshVertexs = stl()[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 

    mvoxel = Stl_to_voxel(config.stl_to_voxel, predict_label=False)
    mvoxel.gen_vox_grid(meshVertexs,config.stl_to_voxel.ray,config.stl_to_voxel.parallel)
    mvoxel.gen_vox_info()
    # full_save_png_name = f"./save/origin_model_png/origin_model_{config.file.model_name}_{timestamp}.png"
    # mvoxel.showVoxel(full_save_png_name)

    # mesh = Voxel_writer(config.voxel_writer, mvoxel.nod_coor_abs, mvoxel.ele_nod)
    # mesh()

    FEA_input = Voxel_to_FEA_input(config.FEA, mvoxel, meshVertexs, predict_label=False)
    full_save_png_name = f"./save/input_model_png/input_model_{config.file.model_name}_{timestamp}.png"
    show_input_fix_force(mvoxel.voxelgrid, FEA_input.fixed_voxel_index, FEA_input.force_voxel_index, full_save_png_name)
    dens_model = Dens_Net()
    dens_model.to('cuda:0')
    disp_model = Disp_Net()
    disp_model.to('cuda:0')
    opt = Dmf_tonn(FEA_input, dens_model, disp_model, config.FEA)
    opt.fit_disp_init(500,1000) # disp_epoch, pinn_epoch
    last_fit_message = opt.fit_to(600)

    # '''
    xPhys_dlX = dens_model(FEA_input.em_dlX)
    xPhys_dlX_full = np.zeros((FEA_input.nely, FEA_input.nelx, FEA_input.nelz))
    xPhys_dlX_full[np.transpose(FEA_input.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
    full_save_png_name = f"./save/output_model_png/output_model_{config.file.model_name}_{timestamp}.png"
    show_output_fix_force(xPhys_dlX_full, FEA_input.fixed_voxel_index, FEA_input.force_voxel_index, full_save_png_name)
    # plot_iso(xPhys_dlX_full)
    full_save_png_name = f"./save/output_model_png/output_mesh_{config.file.model_name}_{timestamp}.png"
    show_mesh(xPhys_dlX_full, FEA_input.fixed_voxel_index, FEA_input.force_voxel_index, full_save_png_name, dens_limits=0.5)

    xPhys_dlX, _, _, _ = opt.dens_filter(xPhys_dlX, FEA_input.em_dlX)
    _, energy_c = opt.pinn.pinn_loss(xPhys_dlX, FEA_input.em_dlX)
    c = torch.mean(energy_c)
    vf = torch.mean(xPhys_dlX)
    print(f'finial VF: {vf.item()}')
    print(f'finial Compliance: {c.item()}')
    xPhys_dlX_full = np.zeros((FEA_input.nely, FEA_input.nelx, FEA_input.nelz))
    xPhys_dlX_full[np.transpose(FEA_input.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
    full_save_png_name = f"./save/output_model_png/output_model_filtered_{config.file.model_name}_{timestamp}.png"
    show_output_fix_force(xPhys_dlX_full, FEA_input.fixed_voxel_index, FEA_input.force_voxel_index, full_save_png_name)
    # plot_iso(xPhys_dlX_full)
    full_save_png_name = f"./save/output_model_png/output_mesh_filtered_{config.file.model_name}_{timestamp}.png"
    show_mesh(xPhys_dlX_full, FEA_input.fixed_voxel_index, FEA_input.force_voxel_index, full_save_png_name, dens_limits=0.5)
    # '''
    
    '''
    mvoxel_predict = Stl_to_voxel(config.stl_to_voxel, predict_label=True)
    mvoxel_predict.gen_vox_grid(meshVertexs,config.stl_to_voxel.ray,config.stl_to_voxel.parallel)
    mvoxel_predict.gen_vox_info()
    full_save_png_name = f"./save/origin_model_png/origin_pridict_model_{config.file.model_name}_{timestamp}.png"
    # mvoxel_predict.showVoxel(full_save_png_name)
    FEA_input_predict = Voxel_to_FEA_input(config.FEA, mvoxel_predict, meshVertexs, predict_label=True)
    full_save_png_name = f"./save/input_model_png/input_pridict_model_{config.file.model_name}_{timestamp}.png"
    show_input_fix_force(mvoxel_predict.voxelgrid, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name)
    
    xPhys_dlX = dens_model(FEA_input_predict.em_dlX)
    xPhys_dlX_full = np.zeros((FEA_input_predict.nely, FEA_input_predict.nelx, FEA_input_predict.nelz))
    xPhys_dlX_full[np.transpose(FEA_input_predict.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
    full_save_png_name = f"./save/output_model_png/output_pridict_model_{config.file.model_name}_{timestamp}.png"
    show_output_fix_force(xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name)
    # plot_iso(xPhys_dlX_full)
    full_save_png_name = f"./save/output_model_png/output_pridict_mesh_{config.file.model_name}_{timestamp}.png"
    show_mesh(xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name, dens_limits=0.5)

    xPhys_dlX, _, _, _ = opt.dens_filter(xPhys_dlX, FEA_input_predict.em_dlX)
    _, energy_c = opt.pinn.pinn_loss(xPhys_dlX, FEA_input_predict.em_dlX)
    c = torch.mean(energy_c)
    vf = torch.mean(xPhys_dlX)
    print(f'finial VF: {vf.item()}')
    print(f'finial Compliance: {c.item()}')
    xPhys_dlX_full = np.zeros((FEA_input_predict.nely, FEA_input_predict.nelx, FEA_input_predict.nelz))
    xPhys_dlX_full[np.transpose(FEA_input_predict.voxelgrid, (1,0,2))] = xPhys_dlX.view(-1).cpu().detach().numpy()
    full_save_png_name = f"./save/output_model_png/output_pridict_model_filtered_{config.file.model_name}_{timestamp}.png"
    show_output_fix_force(xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name)
    # plot_iso(xPhys_dlX_full)
    full_save_png_name = f"./save/output_model_png/output_pridict_mesh_filtered_{config.file.model_name}_{timestamp}.png"
    show_mesh(xPhys_dlX_full, FEA_input_predict.fixed_voxel_index, FEA_input_predict.force_voxel_index, full_save_png_name, dens_limits=0.5)
    '''
    

    full_save_file_name = f"./save/stl/topo_{config.file.model_name}_{timestamp}.stl"
    mesh = save_stl(xPhys_dlX_full, full_save_file_name, dens_limits=0.5)

    full_save_txt_name = f"./save/txt/message_{config.file.model_name}.txt"
    message = [
        f'timestamp: {timestamp}', 
        f'finial VF: {vf.item()}',
        f'finial Compliance: {c.item()}',
        *last_fit_message,
        f'voxel_z_size: {config.stl_to_voxel.voxel_z_size}',
        f'load_posi_ratio: {config.FEA.load_posi_ratio}',
        f'load_dire: {config.FEA.load_dire}',
        f'fixed_voxel_start: {config.FEA.fixed_voxel_start}',
        f'fixed_voxel_orientation: {config.FEA.fixed_voxel_orientation}',
        f'fixed_voxel_num: {config.FEA.fixed_voxel_num}',
        f'target vf: {config.FEA.vf}'
    ]
    save_message_to_txt(full_save_txt_name, message)
"""

if __name__ == '__main__':
    
    mp = main_process()
    mp.get_stl()
    mp.train()
    mp.test_same_ratio()
    # mp.test_multi_ratio()
    mp.save_message()

