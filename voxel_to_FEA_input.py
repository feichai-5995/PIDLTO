import torch
import numpy as np
import math

torch.manual_seed(1)
np.random.seed(1)

class Voxel_to_FEA_input():
    def __init__(self, params, mvoxel, meshVertexs, predict_label=False):

        '''
        if predict_label:
            if mvoxel.voxelgrid.shape[2] == params.nelz * 2:
                [self.nelx, self.nely, self.nelz] = mvoxel.voxelgrid.shape
            else:
                raise ValueError(f'error voxel size !!!')
        else:
            if mvoxel.voxelgrid.shape[2] == params.nelz:
                [self.nelx, self.nely, self.nelz] = mvoxel.voxelgrid.shape
            else:
                raise ValueError(f'error voxel size !!!')
        '''

        self.voxelgrid = mvoxel.voxelgrid
        self.VOXELISE_MP = mvoxel.VOXELISE_MP

        self.voxel_dx = mvoxel.dx
        self.voxel_dy = mvoxel.dy
        self.voxel_dz = mvoxel.dz
        self.half_voxel_dx = mvoxel.dx/2.0 
        self.half_voxel_dy = mvoxel.dy/2.0
        self.half_voxel_dz = mvoxel.dz/2.0


        self.voxel_lx = mvoxel.lx
        self.voxel_ly = mvoxel.ly
        self.voxel_lz = mvoxel.lz

        self.voxel_Nx = mvoxel.Nx
        self.voxel_Ny = mvoxel.Ny
        self.voxel_Nz = mvoxel.Nz
        self.voxel_Ns = mvoxel.Ns
        self.voxel_Nm = mvoxel.Nm
        self.voxel_Ne = mvoxel.voxel_nele

        self.voxel_inside_coor = mvoxel.inside_nod_coor_abs 
        self.voxel_outside_coor = mvoxel.outside_nod_coor_abs
        self.out_nod_ratio = mvoxel.out_nod_ratio

        self.voxel_center_coor = mvoxel.ele_center_coor
        self.inside_voxelgrid = mvoxel.inside_voxelgrid
        self.outside_voxelgrid = mvoxel.outside_voxelgrid  
        self.inside_voxel_center_coor = mvoxel.inside_ele_center_coor
        self.outside_voxel_center_coor = mvoxel.outside_ele_center_coor
        self.outside_ele_ratio = mvoxel.outside_ele_ratio

        self.nor_voxel_scale = mvoxel.nor_voxel_scale
        self.voxel_center = mvoxel.voxel_center 
        self.meshVertexs_normal = mvoxel.meshVertexs_normal 
        self.nor_voxel_volum = mvoxel.nor_voxel_volum 
        
        self.dlX_dx = self.nor_voxel_scale * self.voxel_dx
        self.dlX_dy = self.nor_voxel_scale * self.voxel_dy
        self.dlX_dz = self.nor_voxel_scale * self.voxel_dz

        self.keep_shell = params.keep_shell
        if predict_label:
            self.load_posi_ratio = params.load_posi_ratio_predict
        else:
            self.load_posi_ratio = params.load_posi_ratio
        self.load_dire = params.load_dire

        self.fixed_face_bool = params.fix_face_bool
        if self.fixed_face_bool:
            self.fixed_face = params.fixed_face
            self.fixed_layyer_ratio = params.fixed_layyer_ratio
        else:
            if predict_label:
                self.fixed_voxel_start = params.fixed_voxel_start_predict
                self.fixed_voxel_num = params.fixed_voxel_num_predict
            else:
                self.fixed_voxel_start = params.fixed_voxel_start
                self.fixed_voxel_num = params.fixed_voxel_num
            self.fixed_voxel_orientation = params.fixed_voxel_orientation


        # self.batch_size = params.batch_size
        self.batch_size_mul = params.batch_size_mul
        self.batch_size = int(self.batch_size_mul * self.voxelgrid.sum())


        c_y, c_x, c_z=np.meshgrid(self.voxel_ly[1:]-self.half_voxel_dy, self.voxel_lx[1:]-self.half_voxel_dx, self.voxel_lz[1:]-self.half_voxel_dz, indexing='ij')

        c_y = (c_y - self.voxel_center[1]) * self.nor_voxel_scale
        c_x = (c_x - self.voxel_center[0]) * self.nor_voxel_scale
        c_z = (c_z - self.voxel_center[2]) * self.nor_voxel_scale

        self.dlX = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 1).reshape([-1,3])

        self.embedding = np.transpose(self.voxelgrid, (1,0,2)).reshape(-1,1)
        embedding_indices = np.where(self.embedding.squeeze())[0]
        self.outside_embedding = np.transpose(self.outside_voxelgrid, (1,0,2)).reshape(-1,1).squeeze()[embedding_indices]

        self.em_dlX = self.dlX[self.embedding.squeeze(axis=1)]

        self.em_V = self.voxel_Ne / self.voxel_Ns * self.nor_voxel_volum


        fixed_voxel = np.zeros((self.voxel_Ny, self.voxel_Nx, self.voxel_Nz))
        if self.fixed_face_bool:
            if self.fixed_face == 'x':
                fixed_voxel[:,max(int(self.fixed_layyer_ratio * self.voxel_Nx)-1, 0),:] = 1.0
            elif self.fixed_face == 'y':
                fixed_voxel[max(int(self.fixed_layyer_ratio * self.voxel_Ny)-1, 0),:,:] = 1.0
            elif self.fixed_face == 'z':
                fixed_voxel[:,:,max(int(self.fixed_layyer_ratio * self.voxel_Nz)-1, 0)] = 1.0
            else: 
                raise ValueError(f'wrong fixed_face')
        else:
            start_nelx = min(int(self.fixed_voxel_start[0] * self.voxel_Nx), self.voxel_Nx-1)
            start_nely = min(int(self.fixed_voxel_start[1] * self.voxel_Ny), self.voxel_Ny-1)
            start_nelz = min(int(self.fixed_voxel_start[2] * self.voxel_Nz), self.voxel_Nz-1)
            fixed_voxel_y, fixed_voxel_x, fixed_voxel_z = np.meshgrid(
                np.arange(start_nely, min(start_nely+self.fixed_voxel_orientation[1]*self.fixed_voxel_num[1],self.voxel_Ny), self.fixed_voxel_orientation[1]),
                np.arange(start_nelx, min(start_nelx+self.fixed_voxel_orientation[0]*self.fixed_voxel_num[0],self.voxel_Nx), self.fixed_voxel_orientation[0]),
                np.arange(start_nelz, min(start_nelz+self.fixed_voxel_orientation[2]*self.fixed_voxel_num[2],self.voxel_Nz), self.fixed_voxel_orientation[2]), indexing='ij')
            fixed_voxel_mesh = np.stack((fixed_voxel_y.reshape([-1]),fixed_voxel_x.reshape([-1]),fixed_voxel_z.reshape([-1])),axis = 1).reshape([-1,3])
            fixed_voxel[fixed_voxel_mesh[:, 0], fixed_voxel_mesh[:, 1], fixed_voxel_mesh[:, 2]] = 1.0
        
        self.fixed_voxel_index = np.column_stack(np.where(fixed_voxel == 1.0))[:,[1,0,2]]
        em_fixed_voxel = fixed_voxel.reshape([self.voxel_Ns,1])[self.embedding.squeeze(axis=1)]
        em_dlX_fixed = self.em_dlX[np.where(em_fixed_voxel == 1.0)[0],:]
        print(f'fixed dlX num {em_dlX_fixed.shape[0]}')
        # fixed_voxel = fixed_voxel.reshape([self.nele,1])
        # dlX_fixed = self.dlX[np.where(fixed_voxel == 1.0)[0],:]


        self.force_voxel = np.zeros((self.voxel_Ny,self.voxel_Nx,self.voxel_Nz)) 


        xid = min(int(self.load_posi_ratio[0] * self.voxel_Nx), self.voxel_Nx - 1)
        yid = min(int(self.load_posi_ratio[1] * self.voxel_Ny), self.voxel_Ny - 1)
        zid = min(int(self.load_posi_ratio[2] * self.voxel_Nz), self.voxel_Nz - 1)

        if self.voxelgrid[xid, yid, zid] == False:
            print(f'the force point is not in model')
            true_points = np.argwhere(self.voxelgrid)
            start_array = np.array([xid, yid, zid])
            distances = np.linalg.norm(true_points - start_array, axis=1)
            min_index = np.argmin(distances)
            xid, yid, zid = true_points[min_index]
        self.force_voxel[yid,xid,zid] = 1
        print(f'force voxel id {[xid,yid,zid]}')
        '''
        if self.load_posi_ratio[0] == -1:
            yid = max(int(self.load_posi_ratio[1] * self.voxel_Ny)-1, 0)
            zid = max(int(self.load_posi_ratio[2] * self.voxel_Nz)-1, 0)
            while np.all(self.force_voxel == 0):
                for xid in range(self.voxel_Nx):
                    if self.voxelgrid[xid, yid, zid]:
                        self.force_voxel[yid,xid,zid] = 1
                if yid != 0: yid -= 1
                else: yid += 1
                if zid != 0: zid -= 1
                else: zid += 1
            print(f'force voxel id [-1 {yid,zid}]')
        elif self.load_posi_ratio[1] == -1:
            xid = max(int(self.load_posi_ratio[0] * self.voxel_Nx)-1, 0)
            zid = max(int(self.load_posi_ratio[2] * self.voxel_Nz)-1, 0)
            while np.all(self.force_voxel == 0):
                for yid in range(self.voxel_Ny):
                    if self.voxelgrid[xid, yid, zid]:
                        self.force_voxel[yid,xid,zid] = 1
                if xid != 0: xid -= 1
                else: xid += 1
                if zid != 0: zid -= 1
                else: zid += 1
            print(f'force voxel id [{xid} -1 {zid}]')
        elif self.load_posi_ratio[2] == -1:
            xid = max(int(self.load_posi_ratio[0] * self.voxel_Nx)-1, 0)
            yid = max(int(self.load_posi_ratio[1] * self.voxel_Ny)-1, 0)
            while np.all(self.force_voxel == 0):
                for zid in range(self.voxel_Nz):
                    if self.voxelgrid[xid, yid, zid]:
                        self.force_voxel[yid,xid,zid] = 1
                if xid != 0: xid -= 1
                else: xid += 1
                if yid != 0: yid -= 1
                else: yid += 1
            print(f'force voxel id [{xid,yid} -1]')
        else:
            xid = min(int(self.load_posi_ratio[0] * self.voxel_Nx), self.voxel_Nx - 1)
            yid = min(int(self.load_posi_ratio[1] * self.voxel_Ny), self.voxel_Ny - 1)
            zid = min(int(self.load_posi_ratio[2] * self.voxel_Nz), self.voxel_Nz - 1)

            if self.voxelgrid[xid, yid, zid] == False:
                print(f'the force point is not in model')
                true_points = np.argwhere(self.voxelgrid)
                start_array = np.array([xid, yid, zid])
                distances = np.linalg.norm(true_points - start_array, axis=1)
                min_index = np.argmin(distances)
                xid, yid, zid = true_points[min_index]
            self.force_voxel[yid,xid,zid] = 1
            print(f'force voxel id {[xid,yid,zid]}')
        '''
            
        self.force_voxel_index = np.array([[xid, yid, zid]])
        em_force_voxel = self.force_voxel.reshape([self.voxel_Ns,1])[self.embedding.squeeze(axis=1)]
        em_dlX_force = self.em_dlX[np.where(em_force_voxel == 1)[0],:]


        self.num_em_dlX_fixed = em_dlX_fixed.shape[0]
        self.num_em_dlX_force = em_dlX_force.shape[0]
        

        self.F_vector = torch.tensor([self.load_dire[i] for i in [1, 0, 2]],dtype = torch.float32).to('cuda:0')

        self.em_dlX = torch.tensor(self.em_dlX,dtype=torch.float32, requires_grad=True).to('cuda:0')


        self.em_dlX_fixed = torch.tensor(em_dlX_fixed,dtype=torch.float32).to('cuda:0')
        self.em_dlX_force = torch.tensor(em_dlX_force,dtype=torch.float32).to('cuda:0')

        self.outside_embedding = torch.tensor(self.outside_embedding,dtype=torch.bool, requires_grad=False).to('cuda:0')

    def dlX_disp(self):
        em_dlX_fixed = self.em_dlX_fixed.detach().cpu().numpy()
        em_dlX_force = self.em_dlX_force.detach().cpu().numpy()

        target_point_num = self.batch_size - (self.num_em_dlX_fixed + self.num_em_dlX_force) * int(self.batch_size_mul)
        # target_point_num = self.batch_size - self.num_em_dlX_fixed - self.num_em_dlX_force
        # target_point_num = self.batch_size

        if self.keep_shell:
            real_inside_point_num = self.inside_voxelgrid.sum()
            real_outside_point_num = self.outside_voxelgrid.sum()

            target_outside_point_num = int(self.outside_ele_ratio * target_point_num)
            target_inside_point_num = target_point_num - target_outside_point_num

            gen_inside_voxel_coord = []
            origin_gen_inside_voxel_coord = []
            while True:
                offsets = np.random.uniform(
                    low = - np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                    high = np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                    size=(real_inside_point_num, 3))
                gen_inside_voxel_coord.append(self.inside_voxel_center_coor + offsets)
                origin_gen_inside_voxel_coord.append(self.inside_voxel_center_coor)
                if len(gen_inside_voxel_coord) * len(gen_inside_voxel_coord[0]) >= target_inside_point_num:
                    break
            gen_inside_voxel_coord = np.array(gen_inside_voxel_coord).reshape(-1,3)
            origin_gen_inside_voxel_coord = np.array(origin_gen_inside_voxel_coord).reshape(-1,3)

            inside_center_coord_indices = np.random.choice(gen_inside_voxel_coord.shape[0], size=target_inside_point_num, replace=False)
            inside_center_voxel_coord = gen_inside_voxel_coord[inside_center_coord_indices]
            origin_inside_center_voxel_coord = origin_gen_inside_voxel_coord[inside_center_coord_indices]

            inside_voxel_coord_nor = (inside_center_voxel_coord - self.voxel_center) * self.nor_voxel_scale
            origin_inside_voxel_coord_nor = (origin_inside_center_voxel_coord - self.voxel_center) * self.nor_voxel_scale

            inside_voxel_coord_nor = inside_voxel_coord_nor[:, [1,0,2]]
            origin_inside_voxel_coord_nor = origin_inside_voxel_coord_nor[:, [1,0,2]]

            gen_outside_voxel_coord = []
            origin_gen_outside_voxel_coord = []
            while True:
                offsets = np.random.uniform(
                    low = - np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                    high = np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                    size=(real_outside_point_num, 3))
                gen_outside_voxel_coord.append(self.outside_voxel_center_coor + offsets)
                origin_gen_outside_voxel_coord.append(self.outside_voxel_center_coor)
                if len(gen_outside_voxel_coord) * len(gen_outside_voxel_coord[0]) >= target_outside_point_num:
                    break
            gen_outside_voxel_coord = np.array(gen_outside_voxel_coord).reshape(-1,3)
            origin_gen_outside_voxel_coord = np.array(origin_gen_outside_voxel_coord).reshape(-1,3)

            center_coord_indices = np.random.choice(gen_outside_voxel_coord.shape[0], size=target_outside_point_num, replace=False)
            outside_center_voxel_coord = gen_outside_voxel_coord[center_coord_indices]
            origin_outside_center_voxel_coord = origin_gen_outside_voxel_coord[center_coord_indices]

            outside_voxel_coord_nor = (outside_center_voxel_coord - self.voxel_center) * self.nor_voxel_scale
            origin_outside_voxel_coord_nor = (origin_outside_center_voxel_coord - self.voxel_center) * self.nor_voxel_scale


            outside_voxel_coord_nor = outside_voxel_coord_nor[:, [1,0,2]]
            origin_outside_voxel_coord_nor = origin_outside_voxel_coord_nor[:, [1,0,2]]

            coord_bn = np.concatenate((em_dlX_fixed, em_dlX_force),axis = 0)
            coord = np.concatenate((outside_voxel_coord_nor, inside_voxel_coord_nor, coord_bn),axis = 0)
            origin_coord = np.concatenate((origin_outside_voxel_coord_nor, origin_inside_voxel_coord_nor, coord_bn),axis = 0)

            outside_mask = np.ones(target_outside_point_num, dtype=np.float32) 
            inside_mask = np.zeros(target_inside_point_num, dtype=np.float32)
            bn_mask = np.ones(em_dlX_fixed.shape[0] + em_dlX_force.shape[0], dtype=np.float32)
            outside_coor_mask = np.concatenate((outside_mask, inside_mask, bn_mask))

            coord = torch.tensor(coord, dtype=torch.float32, requires_grad=True).to('cuda:0')
            origin_coord = torch.tensor(origin_coord, dtype=torch.float32, requires_grad=True).to('cuda:0')
            outside_coor_mask = torch.tensor(outside_coor_mask, dtype=torch.bool, requires_grad=False).to('cuda:0')

            indices = torch.randperm(self.batch_size)
            coord = coord[indices]
            origin_coord = origin_coord[indices]
            outside_coor_mask = outside_coor_mask[indices]

        else:
            num_center_coor = self.voxel_center_coor.shape[0]
            gen_voxel_coord = []
            origin_gen_voxel_coord = []
            while True:
                offsets = np.random.uniform(
                    low = - np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                    high = np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                    size=(num_center_coor, 3))
                gen_voxel_coord.append(self.voxel_center_coor + offsets)
                origin_gen_voxel_coord.append(self.voxel_center_coor)
                if len(gen_voxel_coord) * len(gen_voxel_coord[0]) >= target_point_num:
                    break
            gen_voxel_coord = np.array(gen_voxel_coord).reshape(-1,3)
            origin_gen_voxel_coord = np.array(origin_gen_voxel_coord).reshape(-1,3)

            center_coord_indices = np.random.choice(gen_voxel_coord.shape[0], size=target_point_num, replace=False)
            center_voxel_coord = gen_voxel_coord[center_coord_indices]
            origin_center_voxel_coord = origin_gen_voxel_coord[center_coord_indices]

            voxel_coord_nor = (center_voxel_coord - self.voxel_center) * self.nor_voxel_scale
            origin_voxel_coord_nor = (origin_center_voxel_coord - self.voxel_center) * self.nor_voxel_scale

            voxel_coord_nor = voxel_coord_nor[:, [1,0,2]]
            origin_voxel_coord_nor = origin_voxel_coord_nor[:, [1,0,2]]

            coord_bn = np.concatenate((em_dlX_fixed, em_dlX_force),axis = 0)
            # coord_bn_mul = coord_bn
            # for _ in range(self.batch_size_mul-1):
            #     coord_bn_mul = np.concatenate((coord_bn_mul, coord_bn),axis = 0)
            # coord_bn = coord_bn_mul
            coord_bn = np.tile(coord_bn, (int(self.batch_size_mul), 1))
            coord = np.concatenate((voxel_coord_nor, coord_bn),axis = 0)
            origin_coord = np.concatenate((origin_voxel_coord_nor, coord_bn),axis = 0)
            coord = torch.tensor(coord, dtype=torch.float32, requires_grad=True).to('cuda:0')
            origin_coord = torch.tensor(origin_coord, dtype=torch.float32, requires_grad=True).to('cuda:0')

            # coord = torch.tensor(voxel_coord_nor, dtype=torch.float32, requires_grad=True).to('cuda:0')
            # origin_coord = torch.tensor(origin_voxel_coord_nor, dtype=torch.float32, requires_grad=True).to('cuda:0')
            
            indices = torch.randperm(self.batch_size)
            coord = coord[indices]
            origin_coord = origin_coord[indices]
            # 乱序，以免固定坐标都是前几位，导致神经网络训练后自动丢弃批量中前几组数据
            outside_coor_mask = None

        return coord, origin_coord, outside_coor_mask

    def analytical_fixed_BC_disp(self, u, coord, k=10):
        
        
        if self.fixed_face_bool:
            fixed_center = self.fixed_layyer_ratio - 0.5
            if self.fixed_face == 'x':
                offset = coord[:, 1:2] - fixed_center
            elif self.fixed_face == 'y':
                offset = coord[:, 0:1] - fixed_center
            else:
                offset = coord[:, 2:] - fixed_center
            scale = 2 * (1/(1+torch.exp(-20*torch.abs(offset))) - 0.5)
            a = scale.sum()
        else:
            region_bounds_array = []
            nele_num = [self.voxel_Nx, self.voxel_Ny, self.voxel_Nz]

            for i in range(3):
                if self.fixed_voxel_orientation[i] > 0:
                    start_coor = (self.fixed_voxel_start[i] - 0.5) * nele_num[i] / self.voxel_Nm
                    end_coor = start_coor + self.fixed_voxel_num[i] / self.voxel_Nm
                elif self.fixed_voxel_orientation[i] < 0:
                    end_coor = (self.fixed_voxel_start[i] - 0.5) * nele_num[i] / self.voxel_Nm
                    start_coor = end_coor - self.fixed_voxel_num[i] / self.voxel_Nm
                else:
                    raise ValueError(f'error fixed voxel orientation !!!')
                region_bounds_array.append([start_coor, end_coor]) 


            # '''
            scale = torch.zeros(coord.shape[0], device=coord.device)
            region_bounds = {'x':region_bounds_array[0], 'y':region_bounds_array[1], 'z':region_bounds_array[2]}
            for dim, dim_name in enumerate(['y', 'x', 'z']):
                dim_bounds = region_bounds[dim_name]
                dim_coord = coord[:, dim:dim+1]
                lower_bound, upper_bound = dim_bounds

                # lower_trans = ((dim_coord - lower_bound)/(upper_bound - lower_bound))
                lower_trans = dim_coord - lower_bound
                # t1 = (lower_trans > 0.5).sum()
                scale = torch.min(scale, lower_trans.squeeze(1))
                # upper_trans = ((upper_bound - dim_coord)/(upper_bound - lower_bound))
                upper_trans = upper_bound - dim_coord
                # t2 = (upper_trans > 0.5).sum()
                scale = torch.min(scale, upper_trans.squeeze(1))
            scale = (2 * (1/(1+torch.exp(20*scale)) - 0.5))
            a = scale.sum()
            # scale = 1 - torch.sigmoid(k*scale).unsqueeze(1)
            # '''
        
            '''
            scale_list = [] 
            region_bounds = {'x':region_bounds_array[0], 'y':region_bounds_array[1], 'z':region_bounds_array[2]}
            for dim, dim_name in enumerate(['y', 'x', 'z']):
                dim_bounds = region_bounds[dim_name]
                dim_coord = coord[:, dim:dim+1]
                lower_bound, upper_bound = dim_bounds
                
                lower_trans = ((dim_coord - lower_bound)/(upper_bound - lower_bound + 1e-12))
                upper_trans = ((upper_bound - dim_coord)/(upper_bound - lower_bound + 1e-12))

                dim_scale = torch.min(lower_trans.squeeze(dim=1), upper_trans.squeeze(dim=1))
                scale_list.append(dim_scale)
                # scale_flag_list.append(dim_scale)

            scale_tensor = torch.cat(scale_list, dim=1) 
            scale_inside_flag = torch.all(scale_tensor > 0, dim=1, keepdim=True)
            final_scale_list = []
            for i, dim_scale in enumerate(scale_list):
                outside_scale = 2 * torch.sigmoid(-k * dim_scale) - 1
                dim_final_scale = torch.where(
                    scale_inside_flag,
                    torch.zeros_like(dim_scale),
                    outside_scale
                )
                final_scale_list.append(dim_final_scale.squeeze(1))

            scale = torch.stack(final_scale_list, dim=1)  # [n,3]
            '''      
            
            '''
            scale_list = [] 
            region_bounds = {'x':region_bounds_array[0], 'y':region_bounds_array[1], 'z':region_bounds_array[2]}
            for dim, dim_name in enumerate(['y', 'x', 'z']):
                dim_bounds = region_bounds[dim_name]
                dim_coord = coord[:, dim:dim+1]
                lower_bound, upper_bound = dim_bounds
                
                lower_trans = ((dim_coord - lower_bound)/(upper_bound - lower_bound + 1e-12))
                # t1 = (lower_trans > 0.5).sum()
                upper_trans = ((upper_bound - dim_coord)/(upper_bound - lower_bound + 1e-12))
                # t2 = (upper_trans > 0.5).sum()
                # lower_trans = dim_coord - lower_bound
                # upper_trans = upper_bound - dim_coord
                dim_scale = torch.min(lower_trans.squeeze(dim=1), upper_trans.squeeze(dim=1))
                scale_list.append(dim_scale)
                # scale_flag_list.append(dim_scale)
            scale_list_torch = torch.stack(scale_list, dim=1) 
            # scale_flag = 2 * (torch.sigmoid(torch.min(scale_list_torch, dim=1)[0] * k)-0.5)
            scale_flag = torch.where(torch.min(scale_list_torch, dim=1)[0] > 0, torch.ones_like(dim_scale), - torch.ones_like(dim_scale))
            scale = 1 - torch.sigmoid(k * scale_flag.unsqueeze(1) * torch.abs(scale_list_torch))
            # a = scale.sum()
            '''

        # scale = torch.clamp(scale, min=1e-3)

        # fixed_u = u * scale.unsqueeze(1)
        # return fixed_u, scale
        return u, scale
    
    def analytical_fixed_BC_dens(self, xPhys, coord, k=10):
       
        if self.fixed_face_bool:
            fixed_center = self.fixed_layyer_ratio - 0.5
            if self.fixed_face == 'x':
                offset = coord[:, 1:2] - fixed_center
            elif self.fixed_face == 'y':
                offset = coord[:, 0:1] - fixed_center
            else:
                offset = coord[:, 2:] - fixed_center
            scale = 2 * (1 - 1/(1+torch.exp(-20*torch.abs(offset))))
            a = scale.sum()
        else:

            region_bounds_array = []
            nele_num = [self.voxel_Nx, self.voxel_Ny, self.voxel_Nz]

            for i in range(3):
                if self.fixed_voxel_orientation[i] > 0:
                    start_coor = (self.fixed_voxel_start[i] - 0.5) * nele_num[i] / self.voxel_Nm
                    end_coor = start_coor + self.fixed_voxel_num[i] / self.voxel_Nm
                elif self.fixed_voxel_orientation[i] < 0:
                    end_coor = (self.fixed_voxel_start[i] - 0.5) * nele_num[i] / self.voxel_Nm
                    start_coor = end_coor - self.fixed_voxel_num[i] / self.voxel_Nm
                else:
                    raise ValueError(f'error fixed voxel orientation !!!')
                region_bounds_array.append([start_coor, end_coor]) 

            scale = torch.zeros(coord.shape[0], device=coord.device)
            region_bounds = {'x':region_bounds_array[0], 'y':region_bounds_array[1], 'z':region_bounds_array[2]}
            for dim, dim_name in enumerate(['y', 'x', 'z']):
                dim_bounds = region_bounds[dim_name]
                dim_coord = coord[:, dim:dim+1]
                lower_bound, upper_bound = dim_bounds

                # lower_trans = ((dim_coord - lower_bound)/(upper_bound - lower_bound))
                lower_trans = dim_coord - lower_bound
                # t1 = (lower_trans > 0.5).sum()
                scale = torch.min(scale, lower_trans.squeeze(1))
                # upper_trans = ((upper_bound - dim_coord)/(upper_bound - lower_bound))
                upper_trans = upper_bound - dim_coord
                # t2 = (upper_trans > 0.5).sum()
                scale = torch.min(scale, upper_trans.squeeze(1))
            scale = (1.5 - 2 * (1/(1+torch.exp(20*scale))))
            a = scale.sum()

        fixed_xPhys = xPhys * scale

        return fixed_xPhys, scale




    """

    def dlX_disp_origin(self):
        dlX_fixed = self.em_dlX_fixed.detach().cpu().numpy()
        dlX_force = self.em_dlX_force.detach().cpu().numpy()
        domain_xcoord = np.random.uniform(-(self.voxel_Nx-1)/(2*(self.voxel_Nm)),(self.voxel_Nx-1)/(2*(self.voxel_Nm)),(self.batch_size - dlX_fixed.shape[0] - dlX_force.shape[0],1))
        domain_ycoord = np.random.uniform(-(self.voxel_Ny-1)/(2*(self.voxel_Nm)),(self.voxel_Ny-1)/(2*(self.voxel_Nm)),(self.batch_size - dlX_fixed.shape[0] - dlX_force.shape[0],1))
        domain_zcoord = np.random.uniform(-(self.voxel_Nz-1)/(2*(self.voxel_Nm)),(self.voxel_Nz-1)/(2*(self.voxel_Nm)),(self.batch_size - dlX_fixed.shape[0] - dlX_force.shape[0],1))
        # np.random.uniform(low, high ,size)
        domain_coord = np.concatenate((domain_ycoord,domain_xcoord,domain_zcoord),axis = 1)
        coord = np.concatenate((dlX_fixed, dlX_force),axis = 0)
        coord = np.concatenate((coord, domain_coord),axis = 0)
        coord = torch.tensor(coord, dtype=torch.float32, requires_grad=True).to('cuda:0')
        return coord, None

    def dlX_disp_plan1(self):
        em_dlX_fixed = self.em_dlX_fixed.detach().cpu().numpy()
        em_dlX_force = self.em_dlX_force.detach().cpu().numpy()

        target_point_num = self.batch_size - em_dlX_fixed.shape[0] - em_dlX_force.shape[0]
        generate_axis_x_point_num = int(target_point_num * self.nelx / self.nele)
        generate_axis_y_point_num = int(target_point_num * self.nely / self.nele)
        generate_axis_z_point_num = int(target_point_num * self.nelz / self.nele)

        domain_xcoord = np.random.uniform(-(self.nelx-1)/(2*(self.nelm)),(self.nelx-1)/(2*(self.nelm)),(generate_axis_x_point_num,1))
        domain_ycoord = np.random.uniform(-(self.nely-1)/(2*(self.nelm)),(self.nely-1)/(2*(self.nelm)),(generate_axis_y_point_num,1))
        domain_zcoord = np.random.uniform(-(self.nelz-1)/(2*(self.nelm)),(self.nelz-1)/(2*(self.nelm)),(generate_axis_z_point_num,1))
        # np.random.uniform(low, high ,size)
        
        domain_xcoord_grid = np.insert(domain_xcoord,[0, len(domain_xcoord)],[-0.5, 0.5])
        domain_ycoord_grid = np.insert(domain_ycoord,[0, len(domain_ycoord)],[-0.5, 0.5])
        domain_zcoord_grid = np.insert(domain_zcoord,[0, len(domain_zcoord)],[-0.5, 0.5])

        voxcountX = domain_xcoord_grid.size
        voxcountY = domain_ycoord_grid.size
        voxcountZ = domain_zcoord_grid.size
        gridOUTPUT = np.zeros((voxcountX,voxcountY,voxcountZ,3), dtype=bool)
        gridOUTPUT[:,:,:,0] = np.transpose(self.VOXELISE_MP(domain_ycoord_grid,domain_zcoord_grid,domain_xcoord_grid,self.voxel_nor[:,[1,2,0],:]),(2,0,1))
        gridOUTPUT[:,:,:,1] = np.transpose(self.VOXELISE_MP(domain_zcoord_grid,domain_xcoord_grid,domain_ycoord_grid,self.voxel_nor[:,[2,0,1],:]),(1,2,0))
        gridOUTPUT[:,:,:,2] = self.VOXELISE_MP(domain_xcoord_grid,domain_ycoord_grid,domain_zcoord_grid,self.voxel_nor)
        gridOUTPUT = np.sum(gridOUTPUT,axis=3)>=1
        gridOUTPUT = gridOUTPUT[1:-1,:,:]
        gridOUTPUT = gridOUTPUT[:,1:-1,:]
        gridOUTPUT = gridOUTPUT[:,:,1:-1]

        grid_embedding = np.transpose(gridOUTPUT, (1,0,2)).reshape(-1,1).squeeze(axis=1)
        
        domain_coord_y, domain_coord_x, domain_coord_z = np.meshgrid(domain_ycoord, domain_xcoord, domain_zcoord, indexing='ij')
        domain_coord = np.stack([domain_coord_y, domain_coord_x, domain_coord_z], axis=-1).reshape(-1, 3)

        em_domain_coord = domain_coord[grid_embedding]

        coord = np.concatenate((em_dlX_fixed, em_dlX_force),axis = 0)
        coord = np.concatenate((coord, em_domain_coord),axis = 0)
        coord = torch.tensor(coord, dtype=torch.float32, requires_grad=True).to('cuda:0')
        return coord
    
    def dlX_disp_plan2(self):
        em_dlX_fixed = self.em_dlX_fixed.detach().cpu().numpy()
        em_dlX_force = self.em_dlX_force.detach().cpu().numpy()

        target_point_num = self.batch_size - self.num_em_dlX_fixed - self.num_em_dlX_force
        outside_point_num = int(target_point_num * self.out_nod_ratio)
        if outside_point_num > self.voxel_outside_coor.shape[0]:
            outside_point_num = self.voxel_outside_coor.shape[0]
        inside_point_num = target_point_num - outside_point_num

        num_inside_coor = self.voxel_inside_coor.shape[0]
        gen_voxel_coord = []
        origin_gen_voxel_coord = []
        while True:
            offsets = np.random.uniform(
                low = - np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                high = np.array([self.half_voxel_dx, self.half_voxel_dy, self.half_voxel_dz]), 
                size=(num_inside_coor, 3))
            gen_voxel_coord.append(self.voxel_inside_coor + offsets)
            origin_gen_voxel_coord.append(self.voxel_inside_coor)
            if len(gen_voxel_coord) * len(gen_voxel_coord[0]) >= inside_point_num:
            # if len(gen_voxel_coord) * len(gen_voxel_coord[0]) >= target_point_num:
                break
        gen_voxel_coord = np.array(gen_voxel_coord).reshape(-1,3)
        origin_gen_voxel_coord = np.array(origin_gen_voxel_coord).reshape(-1,3)
        inside_indices = np.random.choice(gen_voxel_coord.shape[0], size=inside_point_num, replace=False)
        # inside_indices = np.random.choice(gen_voxel_coord.shape[0], size=target_point_num, replace=False)
        inside_voxel_coord = gen_voxel_coord[inside_indices]
        origin_inside_voxel_coord = origin_gen_voxel_coord[inside_indices]

        outside_indices = np.random.choice(self.voxel_outside_coor.shape[0], size=outside_point_num, replace=False)
        outside_voxel_coord = self.voxel_outside_coor[outside_indices]

        voxel_coord = np.concatenate((inside_voxel_coord, outside_voxel_coord),axis = 0)
        origin_voxel_coord = np.concatenate((origin_inside_voxel_coord, outside_voxel_coord),axis = 0)

        voxel_coord_nor = (voxel_coord - self.voxel_center) * self.voxel_scale
        origin_voxel_coord_nor = (origin_voxel_coord - self.voxel_center) * self.voxel_scale

        # voxel_coord_nor = (inside_voxel_coord - self.voxel_center) * self.voxel_scale
        # origin_voxel_coord_nor = (origin_inside_voxel_coord - self.voxel_center) * self.voxel_scale

        voxel_coord_nor = voxel_coord_nor[:, [1,0,2]]
        origin_voxel_coord_nor = origin_voxel_coord_nor[:, [1,0,2]]

        coord = np.concatenate((em_dlX_fixed, em_dlX_force),axis = 0)
        coord = np.concatenate((coord, voxel_coord_nor),axis = 0)
        origin_coord = np.concatenate((coord, origin_voxel_coord_nor),axis = 0)
        coord = torch.tensor(coord, dtype=torch.float32, requires_grad=True).to('cuda:0')
        origin_coord = torch.tensor(origin_coord, dtype=torch.float32, requires_grad=True).to('cuda:0')
        # indices = torch.randperm(self.batch_size)
        # coord = coord[indices]

        return coord, origin_coord
    """  

    """
    # backup
    def __init__(self, params, mvoxel, mvoxel_predict, meshVertexs):
        if mvoxel.voxelgrid.shape[2] == params.nelz:
            [self.nelx, self.nely, self.nelz] = mvoxel.voxelgrid.shape
            [self.nelx_predict, self.nely_predict, self.nelz_predict] = mvoxel_predict.voxelgrid.shape
        else:
            raise ValueError(f'error voxel size !!!')
        
        self.voxelgrid = mvoxel.voxelgrid
        self.voxel_num = mvoxel.nele
        self.voxelgrid_predict = mvoxel_predict.voxelgrid

        self.VOXELISE_MP = mvoxel.VOXELISE_MP
        [self.voxel_nor, self.voxel_scale, self.voxel_center] = self.meshVertexs_normalize(meshVertexs)
        self.half_voxel_dx = mvoxel.dx / 2
        self.half_voxel_dy = mvoxel.dy / 2
        self.half_voxel_dz = mvoxel.dz / 2
        self.voxel_inside_coor = mvoxel.inside_nod_coor_abs
        self.voxel_outside_coor = mvoxel.outside_nod_coor_abs
        self.out_nod_ratio = mvoxel.out_nod_ratio
        self.voxel_center_coor = mvoxel.ele_center_coor
        
        self.load_posi_ratio = params.load_posi_ratio
        self.load_dire = params.load_dire

        self.fixed_face = params.fixed_face  # 'x'
        self.fixed_layyer_ratio = params.fixed_layyer_ratio  # 0

        self.vf = params.vf
        self.penal = params.penal
        self.E0 = params.E0
        self.nu = params.nu

        self.batch_size = params.batch_size
        self.nele = self.nelx * self.nely * self.nelz
        self.nelm = max(self.nelx, self.nely, self.nelz)

        # c_y, c_x, c_z=np.meshgrid(np.linspace(-(self.nely)/(2*self.nelm), (self.nely)/(2*self.nelm), self.nely),
        #                           np.linspace(-(self.nelx)/(2*self.nelm), (self.nelx)/(2*self.nelm), self.nelx),
        #                           np.linspace(-(self.nelz)/(2*self.nelm), (self.nelz)/(2*self.nelm), self.nelz), indexing='ij')
        c_y, c_x, c_z=np.meshgrid(np.linspace(-(self.nely-1)/(2*self.nelm), (self.nely-1)/(2*self.nelm), self.nely),
                                  np.linspace(-(self.nelx-1)/(2*self.nelm), (self.nelx-1)/(2*self.nelm), self.nelx),
                                  np.linspace(-(self.nelz-1)/(2*self.nelm), (self.nelz-1)/(2*self.nelm), self.nelz), indexing='ij')
        self.dlX = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 1).reshape([-1,3])

        self.embedding = np.transpose(self.voxelgrid, (1,0,2)).reshape(-1,1)

        self.em_dlX = self.dlX[self.embedding.squeeze(axis=1)]

        # self.dlX_dy = self.nely / self.nelm / (self.nely-1)
        # self.dlX_dx = self.nelx / self.nelm / (self.nelx-1)
        # self.dlX_dz = self.nelz / self.nelm / (self.nelz-1)

        self.dlX_dy = 1.0 / self.nelm
        self.dlX_dx = self.dlX_dy
        self.dlX_dz = self.dlX_dy
        
        self.nelm_predict = max(self.nelx_predict,self.nely_predict,self.nelz_predict)
        # c_y, c_x, c_z=np.meshgrid(np.linspace(-(self.nely_predict)/(2*self.nelm_predict),(self.nely_predict)/(2*self.nelm_predict),self.nely_predict),
        #                           np.linspace(-(self.nelx_predict)/(2*self.nelm_predict),(self.nelx_predict)/(2*self.nelm_predict),self.nelx_predict),
        #                           np.linspace(-(self.nelz_predict)/(2*self.nelm_predict),(self.nelz_predict)/(2*self.nelm_predict),self.nelz_predict),indexing='ij')
        c_y, c_x, c_z=np.meshgrid(np.linspace(-(self.nely_predict-1)/(2*self.nelm_predict),(self.nely_predict-1)/(2*self.nelm_predict),self.nely_predict),
                                  np.linspace(-(self.nelx_predict-1)/(2*self.nelm_predict),(self.nelx_predict-1)/(2*self.nelm_predict),self.nelx_predict),
                                  np.linspace(-(self.nelz_predict-1)/(2*self.nelm_predict),(self.nelz_predict-1)/(2*self.nelm_predict),self.nelz_predict),indexing='ij')
        self.dlXSS = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 1).reshape([-1,3])
        
        self.embedding_predict = np.transpose(self.voxelgrid_predict, (1,0,2)).reshape(-1,1)
        self.em_dlXSS = self.dlXSS[self.embedding_predict.squeeze(axis=1)]

        # self.em_dlXSS = self.em_dlX

        self.V = (np.max(self.dlX[:,0])-np.min(self.dlX[:,0]))*(np.max(self.dlX[:,1])-np.min(self.dlX[:,1]))*(np.max(self.dlX[:,2])-np.min(self.dlX[:,2]))

        self.em_V = self.V * self.voxel_num / self.nele


        fixed_voxel = np.zeros((self.nely, self.nelx, self.nelz))
        if self.fixed_face == 'x':
            fixed_voxel[:,min(int(self.fixed_layyer_ratio * self.nelx), self.nelx - 1),:] = 1.0
        elif self.fixed_face == 'y':
            fixed_voxel[min(int(self.fixed_layyer_ratio * self.nely), self.nely - 1),:,:] = 1.0
        elif self.fixed_face == 'z':
            fixed_voxel[:,:,min(int(self.fixed_layyer_ratio * self.nelz), self.nelz - 1)] = 1.0
        else: 
            raise ValueError(f'wrong fixed_face')
        em_fixed_voxel = fixed_voxel.reshape([self.nele,1])[self.embedding.squeeze(axis=1)]
        em_dlX_fixed = self.em_dlX[np.where(em_fixed_voxel == 1.0)[0],:]
        # fixed_voxel = fixed_voxel.reshape([self.nele,1])
        # dlX_fixed = self.dlX[np.where(fixed_voxel == 1.0)[0],:]

        self.force_voxel = np.zeros((self.nely,self.nelx,self.nelz)) 

        if self.load_posi_ratio[0] == -1:
            yid = min(int(self.load_posi_ratio[1] * self.nely), self.nely - 1)
            zid = min(int(self.load_posi_ratio[2] * self.nelz), self.nelz - 1)
            while np.all(self.force_voxel == 0):
                for xid in range(self.nelx):
                    if self.voxelgrid[xid, yid, zid]:
                        self.force_voxel[yid,xid,zid] = 1
                if yid != 0: 
                    yid -= 1
                else:
                    yid += 1
                if zid != 0: 
                    zid -= 1
                else:
                    zid += 1
        elif self.load_posi_ratio[1] == -1:
            xid = min(int(self.load_posi_ratio[0] * self.nelx), self.nelx - 1)
            zid = min(int(self.load_posi_ratio[2] * self.nelz), self.nelz - 1)
            while np.all(self.force_voxel == 0):
                for yid in range(self.nely):
                    if self.voxelgrid[xid, yid, zid]:
                        self.force_voxel[yid,xid,zid] = 1
                if xid != 0: 
                    xid -= 1
                else:
                    xid += 1
                if zid != 0: 
                    zid -= 1
                else:
                    zid += 1
        elif self.load_posi_ratio[2] == -1:
            xid = min(int(self.load_posi_ratio[0] * self.nelx), self.nelx - 1)
            yid = min(int(self.load_posi_ratio[1] * self.nely), self.nely - 1)
            while np.all(self.force_voxel == 0):
                for zid in range(self.nelz):
                    if self.voxelgrid[xid, yid, zid]:
                        self.force_voxel[yid,xid,zid] = 1
                if xid != 0: 
                    xid -= 1
                else:
                    xid += 1
                if yid != 0: 
                    yid -= 1
                else:
                    yid += 1
        else:
            xid = min(int(self.load_posi_ratio[0] * self.nelx), self.nelx - 1)
            yid = min(int(self.load_posi_ratio[1] * self.nely), self.nely - 1)
            zid = min(int(self.load_posi_ratio[2] * self.nelz), self.nelz - 1)

            if self.voxelgrid[xid, yid, zid]:
                self.force_voxel[yid,xid,zid] = 1
            else:
                true_points = np.argwhere(self.voxelgrid)
                start_array = np.array([xid, yid, zid])
                distances = np.linalg.norm(true_points - start_array, axis=1)
                min_index = np.argmin(distances)
                self.force_voxel[true_points[min_index]] = 1
                self.force_voxel[0:2] = self.force_voxel[1], self.force_voxel[0]

        em_force_voxel = self.force_voxel.reshape([self.nele,1])[self.embedding.squeeze(axis=1)]
        em_dlX_force = self.em_dlX[np.where(em_force_voxel == 1)[0],:]
        # force_voxel = self.force_voxel.reshape([self.nele,1])
        # dlX_force = self.dlX[np.where(force_voxel == 1)[0],:]

        self.num_em_dlX_fixed = em_dlX_fixed.shape[0]
        self.num_em_dlX_force = em_dlX_force.shape[0]
        # self.num_dlX_fixed = dlX_fixed.shape[0]
        # self.num_dlX_force = dlX_force.shape[0]
        

        self.F_vector = torch.tensor([self.load_dire[i] for i in [1, 0, 2]],dtype = torch.float32).to('cuda:0')
        self.em_dlX = torch.tensor(self.em_dlX,dtype=torch.float32, requires_grad=True).to('cuda:0')
        self.em_dlXSS = torch.tensor(self.em_dlXSS,dtype=torch.float32, requires_grad=True).to('cuda:0')
        # self.dlX = torch.tensor(self.dlX,dtype=torch.float32, requires_grad=True).to('cuda:0')
        # self.dlXSS = torch.tensor(self.dlXSS,dtype=torch.float32, requires_grad=True).to('cuda:0')

        self.em_dlX_fixed = torch.tensor(em_dlX_fixed,dtype=torch.float32).to('cuda:0')
        self.em_dlX_force = torch.tensor(em_dlX_force,dtype=torch.float32).to('cuda:0')
        # self.dlX_fixed = torch.tensor(dlX_fixed,dtype=torch.float32).to('cuda:0')
        # self.dlX_force = torch.tensor(dlX_force,dtype=torch.float32).to('cuda:0')

        self.iif, self.jf,self.kf = np.meshgrid(np.linspace(0.0,0.0,1),np.linspace(0,self.nely,self.nely+1),np.linspace(0.0,self.nelz,self.nelz+1))  
    """

    """
    @ staticmethod
    def meshVertexs_normalize(meshVertexs):
        meshXmin = meshVertexs[:,0,:].min()
        meshXmax = meshVertexs[:,0,:].max()
        meshYmin = meshVertexs[:,1,:].min()
        meshYmax = meshVertexs[:,1,:].max()
        meshZmin = meshVertexs[:,2,:].min()
        meshZmax = meshVertexs[:,2,:].max()
        scale = 1 / max([meshXmax - meshXmin, meshYmax - meshYmin, meshZmax - meshZmin])
        center = [(meshXmax+meshXmin)/2, (meshYmax+meshYmin)/2, (meshZmax+meshZmin)/2]
        meshVertexs_normal = (meshVertexs - center) * scale
        volum = (meshXmax - meshXmin) * (meshYmax - meshYmin) * (meshZmax - meshZmin) * (scale**3)
        return [meshVertexs_normal, scale, center, volum]
    """