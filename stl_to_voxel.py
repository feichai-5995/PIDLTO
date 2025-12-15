import os
import numpy as np
import time
from scipy.ndimage import binary_erosion, generate_binary_structure
import pyvista as pv

from read_stl import Read_stl
from voxel_writer import Voxel_writer

class Stl_to_voxel:
    '''
    def __init__(self, size, size_base_stl_z, print_message):
        if size_base_stl_z == False and len(size) == 3:
            [self._Nx, self._Ny, self._Nz] = size
        elif size_base_stl_z and isinstance(size,int):
            self._Nz = size
        else:
            raise ValueError(f'error voxel size !!!')
        self._PYPRINT = print_message
        self._size_base_stl_z = size_base_stl_z
        self._nnod = 0
        self._voxel_nele = 0
        self.__meshXmin = 0
        self.__meshXmax = 0
        self.__meshYmin = 0
        self.__meshYmax = 0
        self.__meshZmin = 0
        self.__meshZmax = 0
    '''
    def __init__(self, params, predict_label=False):
        if predict_label:
            params.voxel_z_size = params.voxel_z_size * 2
            params.voxel_xyz_size = [size * 2 for size in params.voxel_xyz_size]

        if params.voxel_base_stl_z and isinstance(params.voxel_z_size,int): 
            self._Nz = params.voxel_z_size
        elif params.voxel_base_stl_z == False and len(params.voxel_xyz_size) == 3: 
            [self._Nx, self._Ny, self._Nz] = params.voxel_xyz_size
        else:
            raise ValueError(f'error voxel size !!!')
        self._PYPRINT = params.print_message
        self._size_base_stl_z = params.voxel_base_stl_z
        self._nnod = 0
        self._voxel_nele = 0
        self.__meshXmin = 0
        self.__meshXmax = 0
        self.__meshYmin = 0
        self.__meshYmax = 0
        self.__meshZmin = 0
        self.__meshZmax = 0

    @property
    def Nx(self):
        return self._Nx
    @property
    def Ny(self):
        return self._Ny
    @property
    def Nz(self):
        return self._Nz
    @property
    def Nm(self):
        return self._Nm

    @property
    def Ns(self):
        return self._Ns

    @property
    def gridCOx(self):
        return self._gridCOx
    @property
    def gridCOy(self):
        return self._gridCOy
    @property
    def gridCOz(self):
        return self._gridCOz

    @property
    def voxelgrid(self):
        return self._voxelgrid
    

    @property
    def nnod(self):
        return self._nnod

    @property
    def voxel_nele(self):
        return self._voxel_nele
    @property
    def ele_nod(self):
        return self._ele_nod

    @property
    def nod_coor(self):
        return self._nod_coor

    @property
    def nod_coor_abs(self):
        return self._nod_coor_abs
    
    @property
    def VOXELISE_MP(self):
        return self._VOXELISE_MP
    @property
    def dx(self):
        return self._dx
    @property
    def dy(self):
        return self._dy
    @property
    def dz(self):
        return self._dz


    @property
    def lx(self):
        return self._lx
    @property
    def ly(self):
        return self._ly
    @property
    def lz(self):
        return self._lz

    @property
    def inside_nod_coor_abs(self):
        return self._inside_nod_coor_abs

    @property
    def outside_nod_coor_abs(self):
        return self._outside_nod_coor_abs

    @property
    def out_nod_ratio(self):
        return self._out_nod_ratio


    @property
    def ele_center_coor(self):
        return self._ele_center_coor

    @property
    def inside_voxelgrid(self):
        return self._inside_voxelgrid

    @property
    def outside_voxelgrid(self):
        return self._outside_voxelgrid

    @property
    def inside_ele_center_coor(self):
        return self._inside_ele_center_coor
    @property
    def outside_ele_center_coor(self):
        return self._outside_ele_center_coor
    @property
    def outside_ele_ratio(self):
        return self._outside_ele_ratio


    @property
    def nor_voxel_scale(self):
        return self._nor_voxel_scale

    @property
    def voxel_center(self):
        return self._voxel_center
    @property
    def meshVertexs_normal(self):
        return self._meshVertexs_normal

    @property
    def nor_voxel_volum(self):
        return self._nor_voxel_volum

    

    def initSize(self, x, y, z):
        self._Nx = x
        self._Ny = y
        self._Nz = z

    def _VOXELISE(self, gridCOx,gridCOy,gridCOz,meshXYZ):
        voxcountX = gridCOx.size
        voxcountY = gridCOy.size
        voxcountZ = gridCOz.size

        gridOUTPUT = np.zeros( (voxcountX,voxcountY,voxcountZ), dtype=int)

        meshXmin = meshXYZ[:,0,:].min()
        meshXmax = meshXYZ[:,0,:].max()
        meshYmin = meshXYZ[:,1,:].min()
        meshYmax = meshXYZ[:,1,:].max()
        meshZmin = meshXYZ[:,2,:].min()
        meshZmax = meshXYZ[:,2,:].max()

        meshXminp = np.where( np.abs(gridCOx-meshXmin)==np.abs(gridCOx-meshXmin).min() )[0][0]
        meshXmaxp = np.where( np.abs(gridCOx-meshXmax)==np.abs(gridCOx-meshXmax).min() )[0][0]
        meshYminp = np.where( np.abs(gridCOy-meshYmin)==np.abs(gridCOy-meshYmin).min() )[0][0]
        meshYmaxp = np.where( np.abs(gridCOy-meshYmax)==np.abs(gridCOy-meshYmax).min() )[0][0]

        if meshXminp > meshXmaxp:
            meshXminp,meshXmaxp = meshXmaxp,meshXminp
        if meshYminp > meshYmaxp:
            meshYminp,meshYmaxp = meshYmaxp,meshYminp

        meshXYZmin = np.min(meshXYZ,axis=2)
        meshXYZmax = np.max(meshXYZ,axis=2)

        correctionLIST = np.zeros((0,2),dtype=int)
        for loopY in range(meshYminp,meshYmaxp+1):
            possibleCROSSLISTy = np.where( (meshXYZmin[:,1]<=gridCOy[loopY]) & (meshXYZmax[:,1]>=gridCOy[loopY]) )[0]

            for loopX in range(meshXminp,meshXmaxp+1):       
                possibleCROSSLIST = possibleCROSSLISTy[ (meshXYZmin[possibleCROSSLISTy,0]<=gridCOx[loopX]) & (meshXYZmax[possibleCROSSLISTy,0]>=gridCOx[loopX]) ]
                if possibleCROSSLIST.size > 0:

                    facetCROSSLIST = np.zeros(0,dtype=int)
                    if possibleCROSSLIST.size > 0:  
                        for loopCHECKFACET in possibleCROSSLIST.flatten():
                            Y1predicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,0])/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]+1e-12))
                            YRpredicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-gridCOx[loopX])/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]+1e-12))
                            
                            if (Y1predicted > meshXYZ[loopCHECKFACET,1,0] and YRpredicted > gridCOy[loopY]) or (Y1predicted < meshXYZ[loopCHECKFACET,1,0] and YRpredicted < gridCOy[loopY]):
                                Y2predicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,1])/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]+1e-12))
                                YRpredicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-gridCOx[loopX])/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]+1e-12))
                            
                                if (Y2predicted > meshXYZ[loopCHECKFACET,1,1] and YRpredicted > gridCOy[loopY]) or (Y2predicted < meshXYZ[loopCHECKFACET,1,1] and YRpredicted < gridCOy[loopY]):
                                    Y3predicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,2])/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]+1e-12))
                                    YRpredicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-gridCOx[loopX])/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]+1e-12))
                                    
                                    if (Y3predicted > meshXYZ[loopCHECKFACET,1,2] and YRpredicted > gridCOy[loopY]) or (Y3predicted < meshXYZ[loopCHECKFACET,1,2] and YRpredicted < gridCOy[loopY]):
                                        facetCROSSLIST = np.insert(facetCROSSLIST,facetCROSSLIST.size,loopCHECKFACET,0)
                        gridCOzCROSS = np.zeros(facetCROSSLIST.shape)
                        for loopFINDZ in facetCROSSLIST:
                            planecoA = meshXYZ[loopFINDZ,1,0]*(meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,2,2]) + meshXYZ[loopFINDZ,1,1]*(meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,2,0]) + meshXYZ[loopFINDZ,1,2]*(meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,2,1])
                            planecoB = meshXYZ[loopFINDZ,2,0]*(meshXYZ[loopFINDZ,0,1]-meshXYZ[loopFINDZ,0,2]) + meshXYZ[loopFINDZ,2,1]*(meshXYZ[loopFINDZ,0,2]-meshXYZ[loopFINDZ,0,0]) + meshXYZ[loopFINDZ,2,2]*(meshXYZ[loopFINDZ,0,0]-meshXYZ[loopFINDZ,0,1]) 
                            planecoC = meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]-meshXYZ[loopFINDZ,1,2]) + meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]-meshXYZ[loopFINDZ,1,0]) + meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]-meshXYZ[loopFINDZ,1,1])
                            planecoD = - meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,1]) - meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,2]) - meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,0])
                            
                            if abs(planecoC) < 1e-14: 
                                planecoC=0
                            else:
                                gridCOzCROSS[facetCROSSLIST==loopFINDZ] = (- planecoD - planecoA*gridCOx[loopX] - planecoB*gridCOy[loopY]) / planecoC

                        gridCOzCROSS = gridCOzCROSS[ (gridCOzCROSS>=meshZmin-1e-12) & (gridCOzCROSS<=meshZmax+1e-12) ]
                        gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
                        gridCOzCROSS = np.unique(gridCOzCROSS)
                        
                        if gridCOzCROSS.size % 2 == 0:  
                            for loopASSIGN in np.arange( 1, (gridCOzCROSS.size/2)+1,dtype=int ):
                                voxelsINSIDE = ((gridCOz>gridCOzCROSS[2*loopASSIGN-2]) & (gridCOz<gridCOzCROSS[2*loopASSIGN-1]))
                                gridOUTPUT[loopX,loopY,voxelsINSIDE] = 1

                        elif gridCOzCROSS.size > 0:
                            correctionLIST = np.insert( correctionLIST, correctionLIST.shape[0], [[loopX,loopY]], axis=0 )

        countCORRECTIONLIST = correctionLIST.shape[0]

        if countCORRECTIONLIST>0:
            if correctionLIST[:,0].min()==1 or correctionLIST[:,0].max()== gridCOx.size or correctionLIST[:,1].min()==1 or correctionLIST[:,1].max()==gridCOy.size:
                gridOUTPUT     = np.pad(gridOUTPUT,((1,1),(1,1),(0,0)),mode='constant')
                correctionLIST = correctionLIST + 1
            
            for loopC in np.arange( 0,countCORRECTIONLIST):
                voxelsforcorrection = np.sum( np.array([gridOUTPUT[correctionLIST[loopC,0]-1,correctionLIST[loopC,1]-1,:] ,\
                                            gridOUTPUT[correctionLIST[loopC,0]-1,correctionLIST[loopC,1],:]   ,\
                                            gridOUTPUT[correctionLIST[loopC,0]-1,correctionLIST[loopC,1]+1,:] ,\
                                            gridOUTPUT[correctionLIST[loopC,0],correctionLIST[loopC,1]-1,:]   ,\
                                            gridOUTPUT[correctionLIST[loopC,0],correctionLIST[loopC,1]+1,:]   ,\
                                            gridOUTPUT[correctionLIST[loopC,0]+1,correctionLIST[loopC,1]-1,:] ,\
                                            gridOUTPUT[correctionLIST[loopC,0]+1,correctionLIST[loopC,1],:]   ,\
                                            gridOUTPUT[correctionLIST[loopC,0]+1,correctionLIST[loopC,1]+1,:] ,\
                                            ]),axis=0 ) 
                voxelsforcorrection = (voxelsforcorrection>=4)
                gridOUTPUT[correctionLIST[loopC,0],correctionLIST[loopC,1],voxelsforcorrection] = 1
            if gridOUTPUT.shape[0]>gridCOx.size or gridOUTPUT.shape[1]>gridCOy.size:
                gridOUTPUT = gridOUTPUT[1:-1,1:-1,:]
        
        return gridOUTPUT

    def _VOXELISE_MP(self, gridCOx,gridCOy,gridCOz,meshXYZ):

        import multiprocessing
        NUM_CORES = multiprocessing.cpu_count()
        from joblib import Parallel, delayed

        voxcountX = gridCOx.size
        voxcountY = gridCOy.size
        voxcountZ = gridCOz.size

        gridshape = (voxcountX,voxcountY,voxcountZ)

        meshXmin = meshXYZ[:,0,:].min()
        meshXmax = meshXYZ[:,0,:].max()
        meshYmin = meshXYZ[:,1,:].min()
        meshYmax = meshXYZ[:,1,:].max()


        meshXminp = np.where( np.abs(gridCOx-meshXmin)==np.abs(gridCOx-meshXmin).min() )[0][0]
        meshXmaxp = np.where( np.abs(gridCOx-meshXmax)==np.abs(gridCOx-meshXmax).min() )[0][0]
        meshYminp = np.where( np.abs(gridCOy-meshYmin)==np.abs(gridCOy-meshYmin).min() )[0][0]
        meshYmaxp = np.where( np.abs(gridCOy-meshYmax)==np.abs(gridCOy-meshYmax).min() )[0][0]

        if meshXminp > meshXmaxp:
            meshXminp,meshXmaxp = meshXmaxp,meshXminp
        if meshYminp > meshYmaxp:
            meshYminp,meshYmaxp = meshYmaxp,meshYminp

        parallel_obj = Parallel(n_jobs=int(NUM_CORES*0.5),verbose=0,backend='loky')
        result = parallel_obj(delayed(self._ray_1direction)(i,gridshape,meshXYZ,gridCOx,gridCOy,gridCOz,meshXminp,meshXmaxp) for i in range(meshYminp,meshYmaxp+1))
        gridOUTPUT = np.array(result).sum(axis=0)
        
        return gridOUTPUT


    def _ray_1direction(self, loopY,gridshape,meshXYZ,gridCOx,gridCOy,gridCOz,meshXminp,meshXmaxp):

        meshXYZmin = np.min(meshXYZ,axis=2)
        meshXYZmax = np.max(meshXYZ,axis=2)
        meshZmin = meshXYZ[:,2,:].min()
        meshZmax = meshXYZ[:,2,:].max()
        gridTemp = np.zeros(gridshape, dtype=int)
        coory = gridCOy[loopY]
        possibleCROSSLISTy = np.where( (meshXYZmin[:,1]<=coory) & (meshXYZmax[:,1]>=coory) )[0]

        correctionLIST = np.zeros((0,2),dtype=int)
        for loopX in range(meshXminp,meshXmaxp+1):  
            coorx = gridCOx[loopX]
            possibleCROSSLIST = possibleCROSSLISTy[ (meshXYZmin[possibleCROSSLISTy,0]<=coorx) & (meshXYZmax[possibleCROSSLISTy,0]>=coorx) ]
            if possibleCROSSLIST.size > 0:
                facetCROSSLIST = np.zeros(0,dtype=int)
                if possibleCROSSLIST.size > 0:
                    for loopCHECKFACET in possibleCROSSLIST.flatten():
                        Y1predicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,0])/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]+1e-12))
                        YRpredicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-coorx)/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]+1e-12))
                        if (Y1predicted > meshXYZ[loopCHECKFACET,1,0] and YRpredicted > coory) or (Y1predicted < meshXYZ[loopCHECKFACET,1,0] and YRpredicted < coory):
                            Y2predicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,1])/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]+1e-12))
                            YRpredicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-coorx)/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]+1e-12))
                            
                            if (Y2predicted > meshXYZ[loopCHECKFACET,1,1] and YRpredicted > coory) or (Y2predicted < meshXYZ[loopCHECKFACET,1,1] and YRpredicted < coory):
                                Y3predicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,2])/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]+1e-12))
                                YRpredicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-coorx)/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]+1e-12))
                                
                                if (Y3predicted > meshXYZ[loopCHECKFACET,1,2] and YRpredicted > coory) or (Y3predicted < meshXYZ[loopCHECKFACET,1,2] and YRpredicted < coory):
                                    facetCROSSLIST = np.insert(facetCROSSLIST,facetCROSSLIST.size,loopCHECKFACET,0)

                    gridCOzCROSS = np.zeros(facetCROSSLIST.shape)
                    for loopFINDZ in facetCROSSLIST:
                        planecoA = meshXYZ[loopFINDZ,1,0]*(meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,2,2]) + meshXYZ[loopFINDZ,1,1]*(meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,2,0]) + meshXYZ[loopFINDZ,1,2]*(meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,2,1])
                        planecoB = meshXYZ[loopFINDZ,2,0]*(meshXYZ[loopFINDZ,0,1]-meshXYZ[loopFINDZ,0,2]) + meshXYZ[loopFINDZ,2,1]*(meshXYZ[loopFINDZ,0,2]-meshXYZ[loopFINDZ,0,0]) + meshXYZ[loopFINDZ,2,2]*(meshXYZ[loopFINDZ,0,0]-meshXYZ[loopFINDZ,0,1]) 
                        planecoC = meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]-meshXYZ[loopFINDZ,1,2]) + meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]-meshXYZ[loopFINDZ,1,0]) + meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]-meshXYZ[loopFINDZ,1,1])
                        planecoD = - meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,1]) - meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,2]) - meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,0])

                        if abs(planecoC) < 1e-14: 
                            planecoC=0
                        else:
                            gridCOzCROSS[facetCROSSLIST==loopFINDZ] = (- planecoD - planecoA*coorx - planecoB*coory) / planecoC

                    gridCOzCROSS = gridCOzCROSS[ (gridCOzCROSS>=meshZmin-1e-12) & (gridCOzCROSS<=meshZmax+1e-12) ]

                    gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
                    gridCOzCROSS = np.unique(gridCOzCROSS)

                    if gridCOzCROSS.size % 2 == 0: 
                        for loopASSIGN in np.arange( 1, (gridCOzCROSS.size/2)+1,dtype=int ):
                            voxelsINSIDE = ((gridCOz>gridCOzCROSS[2*loopASSIGN-2]) & (gridCOz<gridCOzCROSS[2*loopASSIGN-1]))
                            gridTemp[loopX,loopY,voxelsINSIDE] = 1
                    elif gridCOzCROSS.size > 0:
                        correctionLIST = np.insert( correctionLIST, correctionLIST.shape[0], [[loopX,loopY]], axis=0 )
        return gridTemp



    def gen_vox_grid(self, mesharg, ray="xyz", parallel=False):
        if isinstance(mesharg, str):
            stl = Read_stl(mesharg)
            meshXYZ = stl()[0]
        elif isinstance(mesharg, np.ndarray):
            meshXYZ = mesharg

        self.__meshXmin = meshXYZ[:,0,:].min()
        self.__meshXmax = meshXYZ[:,0,:].max()
        self.__meshYmin = meshXYZ[:,1,:].min()
        self.__meshYmax = meshXYZ[:,1,:].max()
        self.__meshZmin = meshXYZ[:,2,:].min()
        self.__meshZmax = meshXYZ[:,2,:].max()

        self._nor_voxel_scale = 1 / max([self.__meshXmax -  self.__meshXmin, self.__meshYmax - self.__meshYmin, self.__meshZmax - self.__meshZmin])
        self._voxel_center = [(self.__meshXmax+self.__meshXmin)/2, (self.__meshYmax+self.__meshYmin)/2, (self.__meshZmax+self.__meshZmin)/2]
        self._meshVertexs_normal = (meshXYZ - self._voxel_center) * self._nor_voxel_scale
        self._nor_voxel_volum = (self.__meshXmax - self.__meshXmin) * (self.__meshYmax - self.__meshYmin) * (self.__meshZmax - self.__meshZmin) * (self._nor_voxel_scale**3)


        if self._size_base_stl_z:
            self._Nx = int(self._Nz * (self.__meshXmax - self.__meshXmin) / (self.__meshZmax - self.__meshZmin))  
            self._Ny = int(self._Nz * (self.__meshYmax - self.__meshYmin) / (self.__meshZmax - self.__meshZmin)) 

        self._Nm = max(self._Nx, self._Ny, self._Nz)
        self._Ns = self._Nx * self._Ny * self._Nz

        voxwidth_x = (self.__meshXmax-self.__meshXmin)/(self._Nx)
        # voxwidth  = (self.__meshXmax-self.__meshXmin)/(self._Nx+1/2)
        self._gridCOx = np.linspace(self.__meshXmin+voxwidth_x/2, self.__meshXmax-voxwidth_x/2, self._Nx)  
        voxwidth_y = (self.__meshYmax-self.__meshYmin)/(self._Ny)
        # voxwidth  = (self.__meshYmax-self.__meshYmin)/(self._Ny+1/2)
        self._gridCOy = np.linspace(self.__meshYmin+voxwidth_y/2, self.__meshYmax-voxwidth_y/2, self._Ny)    
        voxwidth_z = (self.__meshZmax-self.__meshZmin)/(self._Nz)
        # voxwidth  = (self.__meshZmax-self.__meshZmin)/(self._Nz+1/2)
        self._gridCOz = np.linspace(self.__meshZmin+voxwidth_z/2, self.__meshZmax-voxwidth_z/2, self._Nz)  

        gridcheckX = 0
        gridcheckY = 0
        gridcheckZ = 0
        if (self._gridCOx.min()>self.__meshXmin or self._gridCOx.max()<self.__meshXmax):
            if self._gridCOx.min()>self.__meshXmin:
                self._gridCOx = np.insert(self._gridCOx,0,self.__meshXmin,0)
                gridcheckX = gridcheckX + 1
            if self._gridCOx.max()<self.__meshXmax:
                self._gridCOx = np.insert(self._gridCOx,self._gridCOx.size,self.__meshXmax,0)
                gridcheckX = gridcheckX + 2
        if (self._gridCOy.min()>self.__meshYmin or self._gridCOy.max()<self.__meshYmax):
            if self._gridCOy.min()>self.__meshYmin:
                self._gridCOy = np.insert(self._gridCOy,0,self.__meshYmin,0)
                gridcheckY = gridcheckY + 1
            if self._gridCOy.max()<self.__meshYmax:
                self._gridCOy = np.insert(self._gridCOy,self._gridCOy.size,self.__meshYmax,0)
                gridcheckY = gridcheckY + 2
        if (self._gridCOz.min()>self.__meshZmin or self._gridCOz.max()<self.__meshZmax):
            if self._gridCOz.min()>self.__meshZmin:
                self._gridCOz = np.insert(self._gridCOz,0,self.__meshZmin,0)
                gridcheckZ = gridcheckZ + 1
            if self._gridCOz.max()<self.__meshZmax:
                self._gridCOz = np.insert(self._gridCOz,self._gridCOz.size,self.__meshZmax,0)
                gridcheckZ = gridcheckZ + 2

        voxcountX = self._gridCOx.size
        voxcountY = self._gridCOy.size
        voxcountZ = self._gridCOz.size

        gridOUTPUT      = np.zeros( (voxcountX,voxcountY,voxcountZ,len(ray)), dtype=bool)
        countdirections = 0
        
        if parallel:
            if ray.find('x') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(self._VOXELISE_MP(self._gridCOy,self._gridCOz,self._gridCOx,meshXYZ[:,[1,2,0],:]),(2,0,1))
                if self._PYPRINT: print("gridOUTPUT x: ", gridOUTPUT.shape, gridOUTPUT[:,:,:,countdirections].sum())
                countdirections = countdirections + 1
            if ray.find('y') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(self._VOXELISE_MP(self._gridCOz,self._gridCOx,self._gridCOy,meshXYZ[:,[2,0,1],:]),(1,2,0))
                if self._PYPRINT: print("gridOUTPUT y: ", gridOUTPUT.shape, gridOUTPUT[:,:,:,countdirections].sum())
                countdirections = countdirections + 1
            if ray.find('z') + 1:
                gridOUTPUT[:,:,:,countdirections] = self._VOXELISE_MP(self._gridCOx,self._gridCOy,self._gridCOz,meshXYZ)
                if self._PYPRINT: print("gridOUTPUT z: ", gridOUTPUT.shape, gridOUTPUT[:,:,:,countdirections].sum())
                countdirections = countdirections + 1
        else:
            if ray.find('x') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(self._VOXELISE(self._gridCOy,self._gridCOz,self._gridCOx,meshXYZ[:,[1,2,0],:]),(2,0,1))
                if self._PYPRINT: print("gridOUTPUT x: ", gridOUTPUT.shape, gridOUTPUT[:,:,:,countdirections].sum())
                countdirections = countdirections + 1
            if ray.find('y') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(self._VOXELISE(self._gridCOz,self._gridCOx,self._gridCOy,meshXYZ[:,[2,0,1],:]),(1,2,0))
                if self._PYPRINT: print("gridOUTPUT y: ", gridOUTPUT.shape, gridOUTPUT[:,:,:,countdirections].sum())
                countdirections = countdirections + 1
            if ray.find('z') + 1:
                gridOUTPUT[:,:,:,countdirections] = self._VOXELISE(self._gridCOx,self._gridCOy,self._gridCOz,meshXYZ)
                if self._PYPRINT: print("gridOUTPUT z: ", gridOUTPUT.shape, gridOUTPUT[:,:,:,countdirections].sum())
                countdirections = countdirections + 1

        if len(ray)>1:
            gridOUTPUT = np.sum(gridOUTPUT,axis=3)>=int(len(ray)/2)
        if gridcheckX == 1:
            gridOUTPUT = gridOUTPUT[1:,:,:]
            self._gridCOx    = self._gridCOx[1:]
        elif gridcheckX == 2:
            gridOUTPUT = gridOUTPUT[:-1,:,:]
            self._gridCOx    = self._gridCOx[:-1]
        elif gridcheckX == 3:
            gridOUTPUT = gridOUTPUT[1:-1,:,:]
            self._gridCOx    = self._gridCOx[1:-1]
        if gridcheckY == 1:
            gridOUTPUT = gridOUTPUT[:,1:,:]
            self._gridCOy    = self._gridCOy[1:]
        elif gridcheckY == 2:
            gridOUTPUT = gridOUTPUT[:,:-1,:]
            self._gridCOy    = self._gridCOy[:-1]
        elif gridcheckY == 3:
            gridOUTPUT = gridOUTPUT[:,1:-1,:]
            self._gridCOy    = self._gridCOy[1:-1]

        if gridcheckZ == 1:
            gridOUTPUT = gridOUTPUT[:,:,1:]
            self._gridCOz    = self._gridCOz[1:]
        elif gridcheckZ == 2:
            gridOUTPUT = gridOUTPUT[:,:,:-1]
            self._gridCOz    = self._gridCOz[:-1]
        elif gridcheckZ == 3:
            gridOUTPUT = gridOUTPUT[:,:,1:-1]
            self._gridCOz    = self._gridCOz[1:-1]

        if self._PYPRINT:
            print("voxelgrid sum: ", gridOUTPUT.shape, gridOUTPUT.sum())
        
        self._voxelgrid = gridOUTPUT
        return self._voxelgrid

    def gen_vox_info(self):
        if self._PYPRINT:
            print("gen_vox_info start")
        if not isinstance(self._voxelgrid, np.ndarray):
            print("voxelgrid is not generated, run Voxel.VOXELISE first !!!")
            return
        
        ele_vox = self._voxelgrid.transpose((2,1,0))
        self._voxel_nele = len(np.nonzero(ele_vox)[0])
        
        NX, NY, NZ = ele_vox.shape

        origin_nod_vox = np.zeros((NX+1, NY+1, NZ+1))
        cube = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        for c in cube:    
            origin_nod_vox[c[0]:NX+c[0],:,:][:,c[1]:NY+c[1],:][:,:,c[2]:NZ+c[2]] = origin_nod_vox[c[0]:NX+c[0],:,:][:,c[1]:NY+c[1],:][:,:,c[2]:NZ+c[2]] + ele_vox
        
        self._nnod = len(np.nonzero(origin_nod_vox)[0])
                
        print("  voxel_nele:",self._voxel_nele,"nnod:",self._nnod)
        nod_vox = origin_nod_vox > 0

        struct_26 = generate_binary_structure(3, 1)
        self._inside_voxelgrid = binary_erosion(self._voxelgrid, struct_26)
        inside_voxelgrid_num = self._inside_voxelgrid.sum()

        self._outside_voxelgrid = self._inside_voxelgrid ^ self._voxelgrid
        outside_voxelgrid_num = self._outside_voxelgrid.sum()

        self._outside_ele_ratio = outside_voxelgrid_num / self._voxel_nele

        # struct_26 = generate_binary_structure(3, 3) 
        # inside_nod_vox = binary_erosion(nod_vox, struct_26)
        # inside_nnod = inside_nod_vox.sum()

        inside_nod_vox = origin_nod_vox == 8
        inside_nnod = inside_nod_vox.sum()

        outside_nod_vox = nod_vox ^ inside_nod_vox
        outside_nnod = outside_nod_vox.sum()

        self._out_nod_ratio = outside_nnod / self._nnod

        nod_order = np.zeros((NX+1, NY+1, NZ+1))
        nod_order[nod_vox] = range(self._nnod)
        
        cube_g = np.array([[1,1,1],[1,0,1],[1,0,0],[1,1,0],[0,1,1],[0,0,1],[0,0,0],[0,1,0]])
        ele_nod_global = np.zeros((NX*NY*NZ,8), dtype='int32')

        for i in range(8):
            ele_nod_i = nod_order[cube_g[i][0]:NX+cube_g[i][0],:,:][:,cube_g[i][1]:NY+cube_g[i][1],:][:,:,cube_g[i][2]:NZ+cube_g[i][2]]
            ele_nod_global[:,i] = ele_nod_i.flatten()
        self._ele_nod = ele_nod_global[(ele_vox>0).flatten(),:]
        
        nod_x, nod_y, nod_z = np.meshgrid(range(NY+1), range(NX+1), range(NZ+1))
        
        self._nod_coor = np.zeros((self._nnod, 3),dtype='int32')
        self._nod_coor[:,1] = nod_x[nod_vox]
        self._nod_coor[:,2] = nod_y[nod_vox]
        self._nod_coor[:,0] = nod_z[nod_vox]

        inside_nod_coor = np.zeros((inside_nnod, 3),dtype='int32')
        inside_nod_coor[:,1] = nod_x[inside_nod_vox]
        inside_nod_coor[:,2] = nod_y[inside_nod_vox]
        inside_nod_coor[:,0] = nod_z[inside_nod_vox]


        outside_nod_coor = np.zeros((outside_nnod, 3),dtype='int32')
        outside_nod_coor[:,1] = nod_x[outside_nod_vox]
        outside_nod_coor[:,2] = nod_y[outside_nod_vox]
        outside_nod_coor[:,0] = nod_z[outside_nod_vox]

        # self._dx = (self.__meshXmax - self.__meshXmin) / (self._Nx + 0.5)
        # self._dy = (self.__meshYmax - self.__meshYmin) / (self._Ny + 0.5)
        # self._dz = (self.__meshZmax - self.__meshZmin) / (self._Nz + 0.5)
        
        self._dx = (self.__meshXmax - self.__meshXmin) / (self._Nx)
        self._dy = (self.__meshYmax - self.__meshYmin) / (self._Ny)
        self._dz = (self.__meshZmax - self.__meshZmin) / (self._Nz)

        # self._lx = np.arange(self.__meshXmin, self.__meshXmax + self.dx, self.dx)
        # self._ly = np.arange(self.__meshYmin, self.__meshYmax + self.dy, self.dy)
        # self._lz = np.arange(self.__meshZmin, self.__meshZmax + self.dz, self.dz)
        self._lx = np.linspace(self.__meshXmin, self.__meshXmax, self._Nx+1)
        self._ly = np.linspace(self.__meshYmin, self.__meshYmax, self._Ny+1)
        self._lz = np.linspace(self.__meshZmin, self.__meshZmax, self._Nz+1)
        
        self._nod_coor_abs = np.zeros((self._nnod, 3),dtype=float)
        self._nod_coor_abs[:, 0] = self._lx[self._nod_coor[:, 0]]
        self._nod_coor_abs[:, 1] = self._ly[self._nod_coor[:, 1]]
        self._nod_coor_abs[:, 2] = self._lz[self._nod_coor[:, 2]]

        self._inside_nod_coor_abs = np.zeros((inside_nnod, 3),dtype=float)
        self._inside_nod_coor_abs[:, 0] = self._lx[inside_nod_coor[:, 0]]
        self._inside_nod_coor_abs[:, 1] = self._ly[inside_nod_coor[:, 1]]
        self._inside_nod_coor_abs[:, 2] = self._lz[inside_nod_coor[:, 2]]

        self._outside_nod_coor_abs = np.zeros((outside_nnod, 3),dtype=float)
        self._outside_nod_coor_abs[:, 0] = self._lx[outside_nod_coor[:, 0]]
        self._outside_nod_coor_abs[:, 1] = self._ly[outside_nod_coor[:, 1]]
        self._outside_nod_coor_abs[:, 2] = self._lz[outside_nod_coor[:, 2]]

        ele_indices = np.argwhere(self._voxelgrid > 0)
        x_id, y_id, z_id = ele_indices[:, 0], ele_indices[:, 1], ele_indices[:, 2]
        self._ele_center_coor = np.column_stack([
            self.__meshXmin + (x_id + 0.5) * self.dx,
            self.__meshYmin + (y_id + 0.5) * self.dy,
            self.__meshZmin + (z_id + 0.5) * self.dz])
        
        inside_ele_indices = np.argwhere(self._inside_voxelgrid > 0)
        inside_x_id, inside_y_id, inside_z_id = inside_ele_indices[:, 0], inside_ele_indices[:, 1], inside_ele_indices[:, 2]
        self._inside_ele_center_coor = np.column_stack([
            self.__meshXmin + (inside_x_id + 0.5) * self.dx,
            self.__meshYmin + (inside_y_id + 0.5) * self.dy,
            self.__meshZmin + (inside_z_id + 0.5) * self.dz])
        
        outside_ele_indices = np.argwhere(self._outside_voxelgrid > 0)
        outside_x_id, outside_y_id, outside_z_id = outside_ele_indices[:, 0], outside_ele_indices[:, 1], outside_ele_indices[:, 2]
        self._outside_ele_center_coor = np.column_stack([
            self.__meshXmin + (outside_x_id + 0.5) * self.dx,
            self.__meshYmin + (outside_y_id + 0.5) * self.dy,
            self.__meshZmin + (outside_z_id + 0.5) * self.dz])


        if self._PYPRINT:
            print("gen_vox_info finished")
        
        return


    def showVoxel(self, full_save_png_name):

        voxel_grid = self.voxelgrid
        nx, ny, nz = voxel_grid.shape
        
        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Voxel Viewer ({nx}x{ny}x{nz})", font_size=14)
        
        structure = np.ones((3, 3, 3), dtype=bool)
        filled = voxel_grid > 0
        eroded = binary_erosion(filled, structure=structure)
        surface_mask = filled & ~eroded
        x, y, z = np.where(surface_mask)
    
        points = np.zeros((len(x), 3), dtype=np.float32)
        points[:, 0] = x + 0.5
        points[:, 1] = y + 0.5
        points[:, 2] = z + 0.5
        point_cloud = pv.PolyData(points, force_float=False)

        glyphs = point_cloud.glyph(geom=pv.Cube(), scale=False, orient=False)
        
        plotter.add_mesh(glyphs, scalars=None, show_edges=True, edge_color='black',
            color='lightblue', opacity=1.0, pickable=False, name='voxels')
        
        plotter.add_axes(interactive=True)
        plotter.add_floor(color='lightgray', offset=0)
        plotter.show_grid(xtitle='X Index', ytitle='Y Index', ztitle='Z Index',grid='back', location='outer')

        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 0, 1)]
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20
        
        selected_index = (0, 0, 0) 
        highlight_actor = None

        def update_selected_voxel(ix, iy, iz):
            nonlocal selected_index, highlight_actor
 
            if highlight_actor is None:
                highlight_cube = pv.Cube(center=(0.5, 0.5, 0.5), x_length=1.0, y_length=1.0, z_length=1.0)
                highlight_actor = plotter.add_mesh(highlight_cube, color='red', opacity=1.0, style='wireframe',
                    line_width=5, name='highlight', render_lines_as_tubes=True, reset_camera=False, pickable=False)
            plotter.remove_actor('voxel_info')
            highlight_actor.SetPosition(ix, iy, iz)
            plotter.add_text(f"Selected Voxel: ({ix}, {iy}, {iz}) \n Use arrow keys to move: j/l (X), k/i (Y), y/u (Z)", 
                position='lower_right', font_size=12, color='red', name='voxel_info', font='courier', shadow=True)
            selected_index = (ix, iy, iz)
            plotter.iren.interactor.Render()

        update_selected_voxel(0, 0, 0)
        
        def key_callback(obj, event):
            nonlocal selected_index
            
            ix, iy, iz = selected_index
            
            key = obj.GetKeySym()
            
            moved = False
            if key == "j" and ix > 0: ix -= 1; moved = True      
            elif key == "l" and ix < nx-1: ix += 1; moved = True   
            elif key == "k" and iy > 0: iy -= 1; moved = True    
            elif key == "i" and iy < ny-1: iy += 1; moved = True   
            elif key == "y" and iz < nz-1: iz += 1; moved = True   
            elif key == "u" and iz > 0: iz -= 1; moved = True     
            
            if moved: update_selected_voxel(ix, iy, iz)
        
        plotter.iren.interactor.AddObserver("KeyPressEvent", key_callback)
        
        plotter.add_text(
            "Controls: \n - j/l: Move along X-axis \n - k/i: Move along Y-axis \n"
            "- y/u: Move along Z-axis \n - Left drag: Rotate \n - Right drag: Pan \n"
            "- Scroll: Zoom \n - r: Reset view \n - q: Quit",
            position='upper_right', font_size=10, color='gray')
        
        if nx > 0 and ny > 0 and nz > 0:
            x_arrow = pv.Arrow(start=(0,0,0), direction=(nx,0,0))
            y_arrow = pv.Arrow(start=(0,0,0), direction=(0,ny,0))
            z_arrow = pv.Arrow(start=(0,0,0), direction=(0,0,nz))
            
            plotter.add_mesh(x_arrow, color='red', name='x-axis', pickable=False)
            plotter.add_mesh(y_arrow, color='green', name='y-axis', pickable=False)
            plotter.add_mesh(z_arrow, color='blue', name='z-axis', pickable=False)
            
            # plotter.add_point_labels([(nx, 0, 0)], ['X'], text_color='red', font_size=16, pickable=False)
            # plotter.add_point_labels([(0, ny, 0)], ['Y'], text_color='green', font_size=16, pickable=False)
            # plotter.add_point_labels([(0, 0, nz)], ['Z'], text_color='blue', font_size=16, pickable=False)
        
        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved origin model png to: {full_save_png_name}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)

        plotter.enable_depth_peeling()
        plotter.show()
        
    
if __name__=="__main__":
    # import warnings
    # warnings.filterwarnings("ignore")
    
    datapath="./data"
    datapath = os.path.abspath(datapath)+"/"
    dataname = os.path.join(datapath,"hamburger.stl")

    outputpath="./output"
    outputpath = os.path.abspath(outputpath)+"/"
    outputname = os.path.join(outputpath,"hamburger.inp")
    outputname2 = os.path.join(outputpath,"hamburger_2.inp")

    stl = Read_stl(dataname)
    meshVertexs = stl()[0]

    
    print_message = True
    voxel_base_stl_z = True

    if voxel_base_stl_z:
        voxel_shape = 20
    else:    
        voxel_shape = [20,20,20]

    mvoxel = Stl_to_voxel(voxel_shape, voxel_base_stl_z, print_message)
    start=time.time()
    gridOUTPUT = mvoxel.gen_vox_grid(meshVertexs,"xyz",True)
    
    print('Python Voxelise time(P): {:6f}'.format(time.time()-start))
    mvoxel.gen_vox_info()
    mvoxel.showVoxel()

    mesh = Voxel_writer(outputname, 12, mvoxel.nod_coor_abs, mvoxel.ele_nod)
    mesh()



    if voxel_base_stl_z:
        voxel_shape2 = 80
    else:    
        voxel_shape2 = [80,80,80]

    mvoxel2 = Stl_to_voxel(voxel_shape2, voxel_base_stl_z, print_message)
    start=time.time()
    gridOUTPUT2 = mvoxel2.gen_vox_grid(meshVertexs,"xyz",False)
    print('Python Voxelise time(S): {:6f}'.format(time.time()-start))
    mvoxel2.gen_vox_info()
    mvoxel2.showVoxel()

    mesh2 = Voxel_writer(outputname2, 12, mvoxel2.nod_coor_abs, mvoxel2.ele_nod)
    mesh2()