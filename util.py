import torch
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from IPython import display
import numpy as np
from skimage import measure
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import pyvista as pv
from scipy.ndimage import binary_erosion
from scipy import ndimage
import os
from scipy.ndimage import gaussian_filter

class data_process:
    def __init__(self):
        pass

    def show_input_fix_force(self, voxelgrid, fixed_voxel_index, force_voxel_index, full_save_png_name):
        voxel_grid = voxelgrid
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
        plotter.show_grid(xtitle='X Index', ytitle='Y Index', ztitle='Z Index', grid='back', location='outer')
        
        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 1, 0)]
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20
        
        fix_vovel_list = []
        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            if voxel_grid[ix, iy, iz] > 0:
                # name = f'fixed_{ix}_{iy}_{iz}'
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
                fix_vovel_list.append(highlight_cube)
                # plotter.add_mesh(highlight_cube, color='red', opacity=1.0, style='surface',line_width=5, 
                #     name=name, render_lines_as_tubes=True, reset_camera=False, smooth_shading=False)
        if fix_vovel_list:
            fix_voxel_cube = fix_vovel_list[0] if len(fix_vovel_list) == 1 else fix_vovel_list[0].merge(fix_vovel_list[1:])
            plotter.add_mesh(fix_voxel_cube, color='red', opacity=1.0, style='surface',line_width=5, 
                name='fix_voxel_cube', render_lines_as_tubes=True, reset_camera=False, smooth_shading=False)
        
        force_vovel_list = []
        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            if voxel_grid[ix, iy, iz] > 0:
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
                force_vovel_list.append(highlight_cube)
        if force_vovel_list:
            force_voxel_cube = force_vovel_list[0] if len(force_vovel_list) == 1 else force_vovel_list[0].merge(force_vovel_list[1:])
            plotter.add_mesh(force_voxel_cube, color='green', opacity=1.0, style='surface',line_width=5, 
                name='force_voxel_cube', render_lines_as_tubes=True, reset_camera=False, smooth_shading=False)


        plotter.iren.interactor.Render()
                
        legend_entries = [
            ("Voxels", 'blue'),
            ("fixed Voxel", 'red'),
            ("forced Voxel", 'green'),
        ]
        
        plotter.add_legend(
            legend_entries, bcolor=(0.9, 0.9, 0.9), face="r", size=(0.15, 0.1), 
            loc='upper left', border=True)
        
        if nx > 0 and ny > 0 and nz > 0:
            x_arrow = pv.Arrow(start=(0,0,0), direction=(nx,0,0))
            y_arrow = pv.Arrow(start=(0,0,0), direction=(0,ny,0))
            z_arrow = pv.Arrow(start=(0,0,0), direction=(0,0,nz))
            
            plotter.add_mesh(x_arrow, color='red', name='x-axis', pickable=False)
            plotter.add_mesh(y_arrow, color='green', name='y-axis', pickable=False)
            plotter.add_mesh(z_arrow, color='blue', name='z-axis', pickable=False)
            
            plotter.add_point_labels([(nx, 0, 0)], ['X'], text_color='red', font_size=16, pickable=False)
            plotter.add_point_labels([(0, ny, 0)], ['Y'], text_color='green', font_size=16, pickable=False)
            plotter.add_point_labels([(0, 0, nz)], ['Z'], text_color='blue', font_size=16, pickable=False)
        
        plotter.add_text("Controls: \n Left drag: Rotate \n Right drag: Pan \n Scroll: Zoom \n r: Reset view \n q: Quit",
            position='upper_right', font_size=10, color='gray')
        
        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved input model png to: {full_save_png_name}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)

        plotter.enable_depth_peeling()
        plotter.show()

    def show_output_fix_force(self, xPhys_dlX_full, fixed_voxel_index, force_voxel_index, full_save_png_name):
        xPhys_dlX_full = np.swapaxes(xPhys_dlX_full, 0, 1)
        nx, ny, nz = xPhys_dlX_full.shape
        
        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Voxel Viewer ({nx}x{ny}x{nz})", font_size=14)
        
        structure = np.ones((3, 3, 3), dtype=bool)
        filled = xPhys_dlX_full > 0.5
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
        plotter.show_grid(xtitle='X Index', ytitle='Y Index', ztitle='Z Index', grid='back', location='outer')
        
        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 1, 0)]
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20

        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            if xPhys_dlX_full[ix, iy, iz] > 0.5:
                name = f'fixed_{ix}_{iy}_{iz}'
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
                plotter.add_mesh(highlight_cube, color='red', opacity=1.0, style='surface',line_width=5, 
                    name=name, render_lines_as_tubes=True, reset_camera=False, smooth_shading=False)
        
        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            if xPhys_dlX_full[ix, iy, iz] > 0.5:
                name = f'force_{ix}_{iy}_{iz}'
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
                plotter.add_mesh(highlight_cube, color='green', opacity=1.0, style='surface',line_width=5, 
                    name=name, render_lines_as_tubes=True, reset_camera=False, smooth_shading=False)
                plotter.iren.interactor.Render()
                
        legend_entries = [
            ("Voxels", 'blue'),
            ("fixed Voxel", 'red'),
            ("forced Voxel", 'green'),
        ]
        
        plotter.add_legend(
            legend_entries, bcolor=(0.9, 0.9, 0.9), face="r", size=(0.15, 0.1), 
            loc='upper left', border=True)
        
        if nx > 0 and ny > 0 and nz > 0:
            x_arrow = pv.Arrow(start=(0,0,0), direction=(nx,0,0))
            y_arrow = pv.Arrow(start=(0,0,0), direction=(0,ny,0))
            z_arrow = pv.Arrow(start=(0,0,0), direction=(0,0,nz))
            
            plotter.add_mesh(x_arrow, color='red', name='x-axis', pickable=False)
            plotter.add_mesh(y_arrow, color='green', name='y-axis', pickable=False)
            plotter.add_mesh(z_arrow, color='blue', name='z-axis', pickable=False)
            
            plotter.add_point_labels([(nx, 0, 0)], ['X'], text_color='red', font_size=16, pickable=False)
            plotter.add_point_labels([(0, ny, 0)], ['Y'], text_color='green', font_size=16, pickable=False)
            plotter.add_point_labels([(0, 0, nz)], ['Z'], text_color='blue', font_size=16, pickable=False)
        
        plotter.add_text("Controls: \n Left drag: Rotate \n Right drag: Pan \n Scroll: Zoom \n r: Reset view \n q: Quit",
            position='upper_right', font_size=10, color='gray')
        
        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved output model png to: {full_save_png_name}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)
        
        plotter.enable_depth_peeling()
        plotter.show()


    def get_mesh(self, xPhys_dlX_full, target_vf, target_diff=0.1):

        nx, ny, nz = xPhys_dlX_full.shape  
        smooth_field = gaussian_filter(xPhys_dlX_full.astype(float), sigma=0.2)

        padding = np.zeros([nx+2,ny+2,nz+2])
        padding[1:-1, 1:-1, 1:-1] = np.copy(smooth_field)

        low = smooth_field.min()
        high = smooth_field.max()
        total_volume = xPhys_dlX_full.sum()
        while True:
            mid = (low + high) / 2
            verts, faces, normals, values = measure.marching_cubes(padding, level=mid, allow_degenerate=False)
            mesh = pv.PolyData()
            mesh.points = verts
            mesh.faces = np.hstack([np.ones((faces.shape[0], 1)) * 3, faces]).astype(np.int64)
            volume = mesh.volume
            diff = abs(volume - total_volume) / total_volume

            if diff <= target_diff:
                vf = target_vf * volume / total_volume
                return mesh, vf
            else:        
                if volume > total_volume: low = mid
                else: high = mid

    def show_mesh(self, xPhys_dlX_full, fixed_voxel_index, force_voxel_index, full_save_png_name, target_vf, dens_limits=0.5):
        xPhys_dlX_full = np.swapaxes(xPhys_dlX_full, 0, 1)
        ny, nx, nz = xPhys_dlX_full.shape

        mesh, vf = self.get_mesh(xPhys_dlX_full, target_vf)
        
        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Voxel Viewer ({nx}x{ny}x{nz})", font_size=14)
            
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, 
            edge_color='black', opacity=1.0, name='smoothed_mesh')
        
        fixed_spheres = []
        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            if xPhys_dlX_full[ix, iy, iz] > dens_limits:
                center = (ix + 0.5, iy + 0.5, iz + 0.5)
                sphere = pv.Sphere(radius=0.6, center=center)
                fixed_spheres.append(sphere)

        if fixed_spheres:
            fixed_mesh = fixed_spheres[0] if len(fixed_spheres) == 1 else fixed_spheres[0].merge(fixed_spheres[1:])
            plotter.add_mesh(fixed_mesh, color='red', opacity=1.0,name='fixed_points')
        
        force_spheres = []
        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            if xPhys_dlX_full[ix, iy, iz] > dens_limits:
                center = (ix + 0.5, iy + 0.5, iz + 0.5)
                sphere = pv.Sphere(radius=0.6, center=center)
                force_spheres.append(sphere)
        
        if force_spheres:
            force_mesh = force_spheres[0] if len(force_spheres) == 1 else force_spheres[0].merge(force_spheres[1:])
            plotter.add_mesh(force_mesh, color='green', opacity=1.0, name='force_points')
        
        plotter.add_axes(interactive=True)
        plotter.show_grid(xtitle='X Index', ytitle='Y Index', ztitle='Z Index', grid='back', location='outer')
        
        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 1, 0)]
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20
        
        legend_entries = [
            ("Structure", 'lightblue'),
            ("Fixed Points", 'red'),
            ("Force Points", 'green'),
        ]
        
        plotter.add_legend(legend_entries, bcolor=(0.9, 0.9, 0.9), size=(0.15, 0.1), loc='upper left', border=True)
        
        if nx > 0 and ny > 0 and nz > 0:
            plotter.add_point_labels([(nx, 0, 0)], ['X'], text_color='red', font_size=16)
            plotter.add_point_labels([(0, ny, 0)], ['Y'], text_color='green', font_size=16)
            plotter.add_point_labels([(0, 0, nz)], ['Z'], text_color='blue', font_size=16)

        plotter.add_text("Controls: \n Left drag: Rotate \n Right drag: Pan \n Scroll: Zoom \n r: Reset view \n q: Quit",
                            position='upper_right', font_size=10, color='gray')
        
        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved output model png to: {full_save_png_name}, vf={vf}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)

        plotter.enable_depth_peeling()
        plotter.show()

    def save_stl(self, xPhys_dlX_full, save_path, target_vf, dens_limits=0.5):
        xPhys_dlX_full = np.swapaxes(xPhys_dlX_full, 0, 1)
        mesh, vf = self.get_mesh(xPhys_dlX_full, target_vf)
        
        mesh.save(save_path, binary=False)
        print(f"Successfully saved largest connected component to {save_path}, vf={vf}")
        
        return mesh

    def save_message_to_txt(self, savepath, message):
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'a', encoding='utf-8') as f:
            for value in message:
                f.write(f'{value}\n')
            f.write(f'\n\n')
        print(f'Successfully saved message to {savepath}')

    def get_cross_section(self, mesh, full_save_png_name, normal='xyz', origin=None):
        if origin is None:
            origin = mesh.center

        if isinstance(normal, str):
            if normal == 'xyz':
                slice_x = mesh.slice(normal=[1,0,0], origin=origin)
                slice_y = mesh.slice(normal=[0,1,0], origin=origin)
                slice_z = mesh.slice(normal=[0,0,1], origin=origin)
                sections = [
                    (slice_z, 'red', 'XY 平面'),
                    (slice_x, 'green', 'YZ 平面'),
                    (slice_y, 'blue', 'XZ 平面')
                ]
            else:
                normal = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[normal.lower()]
                slice_single = mesh.slice(normal=normal, origin=origin)
                sections = [(slice_single, 'purple', f'垂直于{normal}轴切割')]

        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme(), shape=(2, 2))
        plotter.set_background('white')
        plotter.enable_anti_aliasing('fxaa')
        for i, (section, color, name) in enumerate(sections):
            plotter.subplot(i//2, i%2)
            plotter.add_mesh(mesh, opacity=0.2, color='lightblue')
            plotter.add_mesh(section, color=color, line_width=3)
            plotter.add_title(name)

        plotter.camera_position = 'xy'
        plotter.camera.azimuth = 30  
        plotter.camera.elevation = 20

        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved slice model png to: {full_save_png_name}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)

        plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.1)
        plotter.show()

    def stl_to_mesh(self, stl_file_path):
        mesh = pv.read(stl_file_path)

        print(f"successfully read stl file from: {os.path.basename(stl_file_path)}")
        print(f"points num: {mesh.n_points}")
        print(f"faces num: {mesh.n_faces_strict}")
        
        return mesh

    def show_energy_distribution(self, energy_each_point, xPhys_dlX_full, coord, 
                                fixed_voxel_index, force_voxel_index, cmap='jet'):
        
        energy_each_point = energy_each_point.detach().cpu().numpy()
        coord = coord.detach().cpu().numpy()

        xPhys_dlX_full = np.swapaxes(xPhys_dlX_full, 0, 1)
        nx, ny, nz = xPhys_dlX_full.shape

        structure = np.ones((3, 3, 3), dtype=bool)
        filled = xPhys_dlX_full > 0.5
        eroded = binary_erosion(filled, structure=structure)
        surface_mask = filled & ~eroded
        x, y, z = np.where(surface_mask)

        points = np.stack([x + 0.5, y + 0.5, z + 0.5], axis=1)

        coord_rounded = np.round(coord).astype(int)
        point_indices = []
        for px, py, pz in points.astype(int):
            match = np.where((coord_rounded[:, 0] == px) & 
                            (coord_rounded[:, 1] == py) & 
                            (coord_rounded[:, 2] == pz))[0]
            if len(match) > 0:
                point_indices.append(match[0])
            else:
                point_indices.append(0)
        point_indices = np.array(point_indices)
        point_energy = energy_each_point[point_indices]

        energy_min, energy_max = np.min(point_energy), np.max(point_energy)
        norm_energy = (point_energy - energy_min) / (energy_max - energy_min + 1e-12)

        point_cloud = pv.PolyData(points)
        point_cloud["Energy"] = norm_energy

        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Energy Distribution ({nx}x{ny}x{nz})", font_size=14)

        glyphs = point_cloud.glyph(geom=pv.Cube(), scale=False, orient=False)
        plotter.add_mesh(glyphs, scalars="Energy", cmap=cmap, show_edges=False, opacity=1.0)

        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            name = f'fixed_{ix}_{iy}_{iz}'
            highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
            plotter.add_mesh(highlight_cube, color='red', opacity=1.0, style='surface',
                            line_width=5, name=name, render_lines_as_tubes=True, reset_camera=False)

        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            name = f'force_{ix}_{iy}_{iz}'
            highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
            plotter.add_mesh(highlight_cube, color='green', opacity=1.0, style='surface',
                            line_width=5, name=name, render_lines_as_tubes=True, reset_camera=False)

        plotter.add_axes(interactive=True)
        plotter.add_scalar_bar(title="Normalized Energy", vertical=True, n_labels=5)
        plotter.show_grid(xtitle='X', ytitle='Y', ztitle='Z', grid='back', location='outer')

        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 0, 1)]

        # def save_on_exit(obj, event):
        #     plotter.screenshot(full_save_png_name)
        #     print(f"Saved energy visualization to: {full_save_png_name}")
        # plotter.iren.add_observer('ExitEvent', save_on_exit)
        
        plotter.enable_depth_peeling()
        plotter.show()

    def show_u_reachable(self, xPhys_reachable_full, coord, fixed_voxel_index, force_voxel_index,
                        unreachable_indices_global, reachable_indices_global, cmap='jet'):

        coord = coord.detach().cpu().numpy()

        xPhys_reachable_full = np.swapaxes(xPhys_reachable_full, 0, 1)
        nx, ny, nz = xPhys_reachable_full.shape

        structure = np.ones((3, 3, 3), dtype=bool)
        filled = xPhys_reachable_full > 0.5
        eroded = binary_erosion(filled, structure=structure)
        surface_mask = filled & ~eroded
        x, y, z = np.where(surface_mask)

        points = np.stack([x + 0.5, y + 0.5, z + 0.5], axis=1)
        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Voxel Viewer ({nx}x{ny}x{nz})", font_size=14)

        point_cloud = pv.PolyData(points)
        glyphs = point_cloud.glyph(geom=pv.Cube(), scale=False, orient=False)
        plotter.add_mesh(glyphs, scalars=None, show_edges=True, edge_color='black',
            color='lightblue', opacity=1.0, pickable=False, name='voxels')

        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            if xPhys_reachable_full[ix, iy, iz] > 0.5:
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.04, y_length=1.04, z_length=1.04)
                plotter.add_mesh(highlight_cube, color='red', opacity=1.0, style='surface',
                                line_width=5, render_lines_as_tubes=True, reset_camera=False)

        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            if xPhys_reachable_full[ix, iy, iz] > 0.5:
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.04, y_length=1.04, z_length=1.04)
                plotter.add_mesh(highlight_cube, color='green', opacity=1.0, style='surface',
                                line_width=5, render_lines_as_tubes=True, reset_camera=False)


        if reachable_indices_global is not None and len(reachable_indices_global) > 0:
            centers = np.array(reachable_indices_global) + 0.5
            pc = pv.PolyData(centers)
            glyphs = pc.glyph(geom=pv.Cube(x_length=1.02, y_length=1.02, z_length=1.02), scale=False)
            plotter.add_mesh(glyphs, color='blue', opacity=0.8, show_edges=True,
                             lighting=True, smooth_shading=False, reset_camera=False)

            # for ix, iy, iz in reachable_indices_global:
            #     highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
            #     plotter.add_mesh(highlight_cube, color='blue', opacity=0.6, style='surface',
            #                 line_width=5, render_lines_as_tubes=True, reset_camera=False)

        if unreachable_indices_global is not None and len(reachable_indices_global) > 0:
            centers = np.array(unreachable_indices_global) + 0.5
            pc = pv.PolyData(centers)
            glyphs = pc.glyph(geom=pv.Cube(x_length=1.02, y_length=1.02, z_length=1.02), scale=False)
            plotter.add_mesh(glyphs, color='yellow', opacity=0.8, show_edges=True,
                             lighting=True, smooth_shading=False, reset_camera=False)

            # for ix, iy, iz in unreachable_indices_global:
            #     highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), x_length=1.02, y_length=1.02, z_length=1.02)
            #     plotter.add_mesh(highlight_cube, color='yellow', opacity=0.6, style='surface',
            #                 line_width=5, render_lines_as_tubes=True, reset_camera=False)

        plotter.add_axes(interactive=True)
        plotter.show_grid(xtitle='X', ytitle='Y', ztitle='Z', grid='back', location='outer')

        plotter.camera_position = [(nx * 1.5, ny * 1.5, nz * 1.5), (nx / 2, ny / 2, nz / 2), (0, 1, 0)]
        plotter.camera.azimuth = 60
        plotter.camera.elevation = 30

        plotter.enable_depth_peeling()
        plotter.show()




    """
    def show_output_fix_force_keep_shell(xPhys_dlX_full, fixed_voxel_index, force_voxel_index, full_save_png_name, outside_voxelgrid, outside_opacity=0.5):
        xPhys_dlX_full = np.swapaxes(xPhys_dlX_full, 0, 1)
        nx, ny, nz = xPhys_dlX_full.shape

        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Voxel Viewer ({nx}x{ny}x{nz})", font_size=14)

        structure = np.ones((3, 3, 3), dtype=bool)
        filled = xPhys_dlX_full > 0.5
        eroded = binary_erosion(filled, structure=structure)
        surface_mask = filled & ~eroded
        x, y, z = np.where(surface_mask)
        
        points = np.zeros((len(x), 3), dtype=np.float32)
        points[:, 0] = x + 0.5
        points[:, 1] = y + 0.5
        points[:, 2] = z + 0.5
        
        mask_values = outside_voxelgrid[x, y, z]
        
        inner_points = points[mask_values == 0]
        if len(inner_points) > 0:
            point_cloud_inner = pv.PolyData(inner_points, force_float=False)
            glyphs_inner = point_cloud_inner.glyph(geom=pv.Cube(), scale=False, orient=False)
            plotter.add_mesh(glyphs_inner, scalars=None, show_edges=True, edge_color='black',
                            color='blue', opacity=1.0, pickable=False, name='inner_voxels')

        outer_points = points[mask_values == 1]
        if len(outer_points) > 0:
            point_cloud_outer = pv.PolyData(outer_points, force_float=False)
            glyphs_outer = point_cloud_outer.glyph(geom=pv.Cube(), scale=False, orient=False)
            plotter.add_mesh(glyphs_outer, scalars=None, show_edges=True, edge_color='lightblue',
                            color='lightblue', opacity=outside_opacity, pickable=False, name='outer_voxels')
        
        plotter.add_axes(interactive=True)
        plotter.show_grid(xtitle='X Index', ytitle='Y Index', ztitle='Z Index', grid='back', location='outer')
        
        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 0, 1)]
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20
        
        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            if xPhys_dlX_full[ix, iy, iz] > 0.5:
                name = f'fixed_{ix}_{iy}_{iz}'
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), 
                                        x_length=1.02, y_length=1.02, z_length=1.02)
                plotter.add_mesh(highlight_cube, color='red', opacity=1.0, style='surface',
                                line_width=5, name=name, render_lines_as_tubes=True, 
                                reset_camera=False, smooth_shading=False)
        
        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            if xPhys_dlX_full[ix, iy, iz] > 0.5:
                name = f'force_{ix}_{iy}_{iz}'
                highlight_cube = pv.Cube(center=(ix + 0.5, iy + 0.5, iz + 0.5), 
                                        x_length=1.02, y_length=1.02, z_length=1.02)
                plotter.add_mesh(highlight_cube, color='green', opacity=1.0, style='surface',
                                line_width=5, name=name, render_lines_as_tubes=True, 
                                reset_camera=False, smooth_shading=False)

        legend_entries = [
            ("Voxels (Solid)", 'blue'),
            ("Voxels (Transparent)", (0.5, 0.8, 1.0)),  # 半透明蓝色
            ("fixed Voxel", 'red'),
            ("forced Voxel", 'green'),
        ]

        plotter.add_legend(
            legend_entries, bcolor=(0.9, 0.9, 0.9), face="r", size=(0.15, 0.15), 
            loc='upper left', border=True)
        
        if nx > 0 and ny > 0 and nz > 0:
            x_arrow = pv.Arrow(start=(0,0,0), direction=(nx,0,0))
            y_arrow = pv.Arrow(start=(0,0,0), direction=(0,ny,0))
            z_arrow = pv.Arrow(start=(0,0,0), direction=(0,0,nz))
            
            plotter.add_mesh(x_arrow, color='red', name='x-axis', pickable=False)
            plotter.add_mesh(y_arrow, color='green', name='y-axis', pickable=False)
            plotter.add_mesh(z_arrow, color='blue', name='z-axis', pickable=False)
            
            plotter.add_point_labels([(nx, 0, 0)], ['X'], text_color='red', font_size=16, pickable=False)
            plotter.add_point_labels([(0, ny, 0)], ['Y'], text_color='green', font_size=16, pickable=False)
            plotter.add_point_labels([(0, 0, nz)], ['Z'], text_color='blue', font_size=16, pickable=False)
        
        plotter.add_text("Controls: \n Left drag: Rotate \n Right drag: Pan \n Scroll: Zoom \n r: Reset view \n q: Quit",
            position='upper_right', font_size=10, color='gray')
        
        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved output model png to: {full_save_png_name}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)
        
        plotter.enable_depth_peeling()
        plotter.show()

    def show_mesh_keep_shell(xPhys_dlX_full, fixed_voxel_index, force_voxel_index, full_save_png_name, outside_voxelgrid, dens_limits=0.5, outside_opacity=0.5):

        xPhys_dlX_full = np.swapaxes(xPhys_dlX_full, 0, 1)
        nx, ny, nz = xPhys_dlX_full.shape
            
        padding = np.zeros([nx+2, ny+2, nz+2])
        padding_mask = np.zeros([nx+2, ny+2, nz+2], dtype=float)
        
        padding[1:-1, 1:-1, 1:-1] = np.copy(xPhys_dlX_full)
        padding_mask[1:-1, 1:-1, 1:-1] = np.copy(outside_voxelgrid)
        
        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Voxel Viewer ({nx}x{ny}x{nz})", font_size=14)
        
        inner_padding = np.copy(padding)
        inner_padding[padding_mask >= 0.5] = 0.0
        
        inner_verts, inner_faces, _, _ = measure.marching_cubes(
            inner_padding, level=dens_limits, 
            allow_degenerate=False, gradient_direction='descent'
        )
        
        inner_mesh = pv.PolyData()
        inner_mesh.points = inner_verts
        inner_mesh.faces = np.hstack([np.ones((inner_faces.shape[0], 1)) * 3, inner_faces]).astype(np.int64)
        
        outer_padding = np.copy(padding)
        outer_padding[padding_mask < 0.5] = 0.0
        
        outer_verts, outer_faces, _, _ = measure.marching_cubes(
            outer_padding, level=dens_limits, 
            allow_degenerate=False, gradient_direction='descent'
        )
        
        outer_mesh = pv.PolyData()
        outer_mesh.points = outer_verts
        outer_mesh.faces = np.hstack([np.ones((outer_faces.shape[0], 1)) * 3, outer_faces]).astype(np.int64)
        
        plotter.add_mesh(inner_mesh, color='blue', show_edges=True, 
                        edge_color='black', opacity=1.0, name='inner_mesh')
        
        plotter.add_mesh(outer_mesh, color='lightblue', show_edges=True, 
                        edge_color='lightblue', opacity=outside_opacity, name='outer_mesh')
        
        boundary_mask = (padding_mask >= 0.4) & (padding_mask <= 0.6)

        boundary_padding = np.copy(padding)
        boundary_padding[~boundary_mask] = 0.0
        
        if np.any(boundary_mask):
            boundary_verts, boundary_faces, _, _ = measure.marching_cubes(
                boundary_padding, level=dens_limits, 
                allow_degenerate=False, gradient_direction='descent'
            )
            
            boundary_mesh = pv.PolyData()
            boundary_mesh.points = boundary_verts
            boundary_mesh.faces = np.hstack([np.ones((boundary_faces.shape[0], 1)) * 3, boundary_faces]).astype(np.int64)
            
            plotter.add_mesh(boundary_mesh, color='lightblue', show_edges=True, 
                            edge_color='black', opacity=0.75, name='boundary_mesh')
        
        fixed_spheres = []
        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            if xPhys_dlX_full[ix, iy, iz] > dens_limits:
                center = (ix + 1.5, iy + 1.5, iz + 1.5)
                sphere = pv.Sphere(radius=0.6, center=center)
                fixed_spheres.append(sphere)

        if fixed_spheres:
            fixed_mesh = fixed_spheres[0] if len(fixed_spheres) == 1 else fixed_spheres[0].merge(fixed_spheres[1:])
            plotter.add_mesh(fixed_mesh, color='red', opacity=1.0, name='fixed_points')

        force_spheres = []
        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            if xPhys_dlX_full[ix, iy, iz] > dens_limits:
                center = (ix + 1.5, iy + 1.5, iz + 1.5)
                sphere = pv.Sphere(radius=0.6, center=center)
                force_spheres.append(sphere)
        
        if force_spheres:
            force_mesh = force_spheres[0] if len(force_spheres) == 1 else force_spheres[0].merge(force_spheres[1:])
            plotter.add_mesh(force_mesh, color='green', opacity=1.0, name='force_points')

        plotter.add_axes(interactive=True)
        plotter.show_grid(xtitle='X Index', ytitle='Y Index', ztitle='Z Index', grid='back', location='outer')
        
        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 0, 1)]
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20
        
        legend_entries = [
            ("Core Structure", 'lightblue'),
            ("Boundary Layer", (0.7, 0.85, 1.0)),
            ("Outer Layer", (0.5, 0.5, 1.0)),
            ("Fixed Points", 'red'),
            ("Force Points", 'green'),
        ]
        
        plotter.add_legend(legend_entries, bcolor=(0.9, 0.9, 0.9), size=(0.15, 0.2), loc='upper left', border=True)
        if nx > 0 and ny > 0 and nz > 0:
            plotter.add_point_labels([(nx, 0, 0)], ['X'], text_color='red', font_size=16)
            plotter.add_point_labels([(0, ny, 0)], ['Y'], text_color='green', font_size=16)
            plotter.add_point_labels([(0, 0, nz)], ['Z'], text_color='blue', font_size=16)
        
        plotter.add_text("Controls: \n Left drag: Rotate \n Right drag: Pan \n Scroll: Zoom \n r: Reset view \n q: Quit",
                        position='upper_right', font_size=10, color='gray')
        
        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved output model png to: {full_save_png_name}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)

        plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.1)
        plotter.show()

    def show_mesh_keep_shell_plan2(xPhys_dlX_full, fixed_voxel_index, force_voxel_index, full_save_png_name, outside_voxelgrid, dens_limits=0.5, outside_opacity=0.5):
        xPhys_dlX_full = np.swapaxes(xPhys_dlX_full, 0, 1)
        nx, ny, nz = xPhys_dlX_full.shape

        padding = np.zeros([nx+2, ny+2, nz+2])
        padding_mask = np.zeros([nx+2, ny+2, nz+2], dtype=bool)
        
        padding[1:-1, 1:-1, 1:-1] = np.copy(xPhys_dlX_full)
        padding_mask[1:-1, 1:-1, 1:-1] = np.copy(outside_voxelgrid)
        
        plotter = pv.Plotter(window_size=[1000, 800], theme=pv.themes.DocumentTheme())
        plotter.enable_anti_aliasing('fxaa')
        plotter.add_title(f"Voxel Viewer ({nx}x{ny}x{nz})", font_size=14)
        
        verts, faces, normals, values = measure.marching_cubes(padding, level=dens_limits, allow_degenerate=False, gradient_direction='descent')
        
        voxel_indices = np.floor(verts).astype(int)
        
        vertex_mask = np.zeros(len(verts), dtype=bool)
        for i, idx in enumerate(voxel_indices):
            ix, iy, iz = np.clip(idx, [1, 1, 1], [nx, ny, nz])
            vertex_mask[i] = padding_mask[ix, iy, iz]
        
        inner_faces = []
        for face in faces:
            if all(not vertex_mask[vertex_idx] for vertex_idx in face):
                inner_faces.append(face)
        
        if inner_faces:
            inner_faces = np.array(inner_faces)
            inner_mesh = pv.PolyData()
            inner_mesh.points = verts
            inner_mesh.faces = np.hstack([np.ones((inner_faces.shape[0], 1)) * 3, inner_faces]).astype(np.int64)
            plotter.add_mesh(inner_mesh, color='blue', show_edges=True, 
                            edge_color='black', opacity=1.0, name='inner_mesh')
        
        outer_faces = []
        for face in faces:
            if any(vertex_mask[vertex_idx] for vertex_idx in face):
                outer_faces.append(face)
        
        if outer_faces:
            outer_faces = np.array(outer_faces)
            outer_mesh = pv.PolyData()
            outer_mesh.points = verts
            outer_mesh.faces = np.hstack([np.ones((outer_faces.shape[0], 1)) * 3, outer_faces]).astype(np.int64)
            plotter.add_mesh(outer_mesh, color='lightblue', show_edges=True, 
                            edge_color='lightblue', opacity=outside_opacity, name='outer_mesh')
        fixed_spheres = []
        for fix_voxel in fixed_voxel_index:
            ix, iy, iz = map(int, fix_voxel)
            if xPhys_dlX_full[ix, iy, iz] > dens_limits:
                center = (ix + 1.5, iy + 1.5, iz + 1.5)
                sphere = pv.Sphere(radius=0.6, center=center)
                fixed_spheres.append(sphere)

        if fixed_spheres:
            fixed_mesh = fixed_spheres[0] if len(fixed_spheres) == 1 else fixed_spheres[0].merge(fixed_spheres[1:])
            plotter.add_mesh(fixed_mesh, color='red', opacity=1.0, name='fixed_points')
        
        force_spheres = []
        for force_voxel in force_voxel_index:
            ix, iy, iz = map(int, force_voxel)
            if xPhys_dlX_full[ix, iy, iz] > dens_limits:
                center = (ix + 1.5, iy + 1.5, iz + 1.5)
                sphere = pv.Sphere(radius=0.6, center=center)
                force_spheres.append(sphere)
        
        if force_spheres:
            force_mesh = force_spheres[0] if len(force_spheres) == 1 else force_spheres[0].merge(force_spheres[1:])
            plotter.add_mesh(force_mesh, color='green', opacity=1.0, name='force_points')
        
        plotter.add_axes(interactive=True)
        plotter.show_grid(xtitle='X Index', ytitle='Y Index', ztitle='Z Index', grid='back', location='outer')

        plotter.camera_position = [(nx*1.5, ny*1.5, nz*1.5), (nx/2, ny/2, nz/2), (0, 0, 1)]
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20
        
        legend_entries = [
            ("Structure (Solid)", 'lightblue'),
            ("Structure (Transparent)", (0.5, 0.8, 1.0)),  # 半透明蓝色
            ("Fixed Points", 'red'),
            ("Force Points", 'green'),
        ]
        
        plotter.add_legend(legend_entries, bcolor=(0.9, 0.9, 0.9), size=(0.15, 0.15), loc='upper left', border=True)

        if nx > 0 and ny > 0 and nz > 0:
            plotter.add_point_labels([(nx, 0, 0)], ['X'], text_color='red', font_size=16)
            plotter.add_point_labels([(0, ny, 0)], ['Y'], text_color='green', font_size=16)
            plotter.add_point_labels([(0, 0, nz)], ['Z'], text_color='blue', font_size=16)
        
        plotter.add_text("Controls: \n Left drag: Rotate \n Right drag: Pan \n Scroll: Zoom \n r: Reset view \n q: Quit",
                        position='upper_right', font_size=10, color='gray')
        
        def save_on_exit(obj, event):
            plotter.screenshot(full_save_png_name)
            print(f"Successfully saved output model png to: {full_save_png_name}")

        plotter.iren.add_observer('ExitEvent', save_on_exit)

        plotter.enable_depth_peeling()
        plotter.show()

"""
