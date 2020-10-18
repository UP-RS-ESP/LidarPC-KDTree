#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:36:08 2019
Updated on Oct-14-2020

@author: bodo.bookhagen@uni-potsdam.de


Next, generate the python/conda environment:
conda create -y -n PC_py38 -c rapidsai-nightly -c nvidia -c anaconda -c conda-forge -c defaults \
ipython spyder python=3.8 rapids=0.16 cudatoolkit=11.0 cuspatial \
  gdal=3 numpy scipy dask h5py pandas pytables hdf5 cython matplotlib tabulate \
  scikit-learn pyflann cyflann scikit-image opencv ipywidgets scikit-learn laszip liblas
conda activate PC_py38
pip install tables
pip install laspy

#Install only CUDA-Dataframe
conda create -y -n cuDF -c rapidsai-nightly -c nvidia -c conda-forge \
    -c defaults rapids=0.16 python=3.8 cudatoolkit=11.0 ipython numpy scipy

System:
AMD Ryzen 9 3900X 12-Core Processor
GeForce RTX 2080 SUPER with 3072 CUDA cores (NVIDIA Corporation TU104 [GeForce RTX 2080 SUPER] (rev a1))
"""

import argparse, numpy as np, pandas as pd, timeit, os, subprocess
import matplotlib.cm as cm

def write_LAS(pc_pc_xyz, v, output_las_fn, input_las_fn, cmap=cm.terrain, rescale='none'):
    from laspy.file import File
    from skimage import exposure
    import copy

    inFile = File(input_las_fn, mode='r')

    #normalize input and generate colors for height using colormap
    #stretch to 10-90th percentile
    #v_1090p = np.percentile(v, [10, 90])
    #stretch to 2-98th percentile
    v_0298p = np.percentile(v, [2, 98])
    if rescale == 'none':
        v_rescale = exposure.rescale_intensity(v, in_range=(v_0298p[0], v_0298p[1]))
    elif rescale == 'median':
        bounds = np.round(np.median(np.abs(v_0298p)), decimals=2)
        v_rescale = exposure.rescale_intensity(v, in_range=(-bounds, bounds))

    colormap_terrain = cmap
    rgb = colormap_terrain(v_rescale)
    #remove last column - alpha value
    rgb = (rgb[:, :3] * (np.power(2,16)-1)).astype('uint16')
    outFile = File(output_las_fn, mode='w', header=inFile.header)
    new_header = copy.copy(outFile.header)
    #setting some variables
    new_header.created_year = datetime.datetime.now().year
    new_header.created_day = datetime.datetime.now().timetuple().tm_yday
    new_header.x_max = pc_pc_xyz[:,0].max()
    new_header.x_min = pc_pc_xyz[:,0].min()
    new_header.y_max = pc_pc_xyz[:,1].max()
    new_header.y_min = pc_pc_xyz[:,1].min()
    new_header.z_max = pc_pc_xyz[:,2].max()
    new_header.z_min = pc_pc_xyz[:,2].min()
    new_header.point_records_count = pc_pc_xyz.shape[0]
    new_header.point_return_count = 0
    outFile.header.count = v.shape[0]
    new_header.scale=inFile.header.scale
    new_header.offset=inFile.header.offset
    outFile.X = (pc_pc_xyz[:,0]-inFile.header.offset[0])/inFile.header.scale[0]
    outFile.Y = (pc_pc_xyz[:,1]-inFile.header.offset[1])/inFile.header.scale[1]
    outFile.Z = (pc_pc_xyz[:,2]-inFile.header.offset[2])/inFile.header.scale[2]
    outFile.Red = rgb[:,0]
    outFile.Green = rgb[:,1]
    outFile.Blue = rgb[:,2]
    outFile.close()

def load_LAS(las_fname, dtype='float32'):
    """
    Load LAS or LAZ file (only coordinates) and return pc_xyz and xy vectors. Converts float64 to float32 by default, unless you set dtype='float64'
    """
    from laspy.file import File

    inFile = File(las_fname, mode='r')
    pc_pc_xyz = np.vstack((inFile.get_x()*inFile.header.scale[0]+inFile.header.offset[0], inFile.get_y()*inFile.header.scale[1]+inFile.header.offset[1], inFile.get_z()*inFile.header.scale[2]+inFile.header.offset[2])).transpose()

    #setting datatype to float32 to save memory.
    if dtype == 'float32':
        pc_pc_xyz = pc_pc_xyz.astype('float32')
    return pc_pc_xyz

def cmdLineParser():
    parser = argparse.ArgumentParser(description='Compare KDTree algorithms for lidar or SfM PointClouds (PC). B. Bookhagen (bodo.bookhagen@uni-potsdam.de), Aug 2019.')
    parser.add_argument('-i', '--inlas', type=str, default='Pozo_WestCanada_clg.laz',  help='LAS/LAZ file with point-cloud data.')
    return parser.parse_args()

def pc_generate_cyflannKDTree(pc_xyz):
    #conda install -y -c conda-forge cyflann
    try:
        import cyflann
    except ImportError:
        raise pc_generate_cyflannKDTree("pyflann not installed.")
    cyflann.set_distance_type('euclidean')
    pc_xyz_cyflannKDTree_tree = cyflann.FLANNIndex()
    pc_xyz_cyflannKDTree_tree.build_index(pc_xyz, algorithm='kdtree_single')
    return pc_xyz_cyflannKDTree_tree

def pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=10):
    pc_cyflannKDTree_id, pc_cyflannKDTree_distance = pc_xyz_cyflannKDTree_tree.nn_index(pc_xyz, k)
    return pc_cyflannKDTree_distance, pc_cyflannKDTree_id


def pc_density_sphere(d, k=10):
    dens =  k / np.pi / d[:, -1]**2
    return dens

def pc_density_ellipsoid(pc_xyz, pc_ids, k=10):
    #estimate density in x, y, z directions using an ellipsoid and returning lengthening ratios in x, y, z directions

    x = np.max(pc_xyz[pc_ids[:,1::],0],axis=1) - np.min(pc_xyz[pc_ids[:,1::],0],axis=1)
    y = np.max(pc_xyz[pc_ids[:,1::],1],axis=1) - np.min(pc_xyz[pc_ids[:,1::],1],axis=1)
    z = np.max(pc_xyz[pc_ids[:,1::],2],axis=1) - np.min(pc_xyz[pc_ids[:,1::],2],axis=1)
    dens_ellipsoid = k / (4/3 * np.pi * x * y * z )
    pts_x_y_ratio = x / y #values < 1 indicate X axis is compressed (and Y axis is longer)
    pts_x_z_ratio = x / z #values < 1 indicate X axis is compressed (and Z axis is longer), values > 1 indicate X axis is longer
    pts_y_z_ratio = y / z #values < 1 indicate X axis is compressed (and Z axis is longer), values > 1 indicate X axis is longer
    return dens_ellipsoid, pts_x_y_ratio, pts_x_z_ratio, pts_y_z_ratio

def pc_knn_geometry(pc_xyz, pc_ids, k=10):
    #estimate position of current point with respect to its neighbors.
    #estimate density in x, y, z directions using an ellipsoid and returning lengthening ratios in x, y, z directions

if __name__ == '__main__':
    inps = cmdLineParser()
# Testing
    inps = argparse.ArgumentParser(description='Compare KDTree algorithms for lidar or SfM PointClouds (PC). Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de), Aug 2019.')
    inps.inlas = 'Pozo_WestCanada_clg.laz'
    inps.inlas = '/home/bodo/Dropbox/CampusGolm-Lidar/Golm_PC_Milan_MavicPro2_Inspire2/Golm_May06_2018_Milan_UTM33N_WGS84_6digit_cl_clip.las'
    inps.nr_of_repetitions = 5
    #AMD Ryzen 9 3900X 12-Core Processor
    inps.hdf_filename = 'AMD_Ryzen9_3900X.hdf'
    inps.csv_filename = 'AMD_Ryzen9_3900X_repetitions5.csv'
    inps.logdir = 'log'
    inps.write_to_LASFILES = True

    log_dir = inps.logdir

    print('Loading input file: %s... '%inps.inlas, end='', flush=True)
    pc_xyz = load_LAS(inps.inlas, dtype='float32')
    print('loaded %s points'%"{:,}".format(pc_xyz.shape[0]))

    pc_xyz_cyflannKDTree_tree = pc_generate_cyflannKDTree(pc_xyz)
    pc_cyflannKDTree_distance, pc_cyflannKDTree_id = pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=5)
    pc_dens_sphere_k5 = pc_density_sphere(pc_cyflannKDTree_distance, k=5)
    pc_dens_ellipsoid_k5, pts_x_y_ratio_k5, pts_x_z_ratio_k5, pts_y_z_ratio_k5 = pc_density_ellipsoid(pc_xyz, pc_cyflannKDTree_id, k=5)

    pc_cyflannKDTree_distance, pc_cyflannKDTree_id = pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=10)
    pc_dens_sphere_k10 = pc_density_sphere(pc_cyflannKDTree_distance, k=10)
    pc_dens_ellipsoid_k10, pts_x_y_ratio_k10, pts_x_z_ratio_k10, pts_y_z_ratio_k10 = pc_density_ellipsoid(pc_xyz, pc_cyflannKDTree_id, k=10)

    pc_cyflannKDTree_distance, pc_cyflannKDTree_id = pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=50)
    pc_dens_sphere_k50 = pc_density_sphere(pc_cyflannKDTree_distance, k=50)
    pc_dens_ellipsoid_k50, pts_x_y_ratio_k50, pts_x_z_ratio_k50, pts_y_z_ratio_k50 = pc_density_ellipsoid(pc_xyz, pc_cyflannKDTree_id, k=50)

    pc_cyflannKDTree_distance, pc_cyflannKDTree_id = pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=100)
    pc_dens_sphere_k100 = pc_density_sphere(pc_cyflannKDTree_distance, k=100)
    pc_dens_ellipsoid_k100, pts_x_y_ratio_k100, pts_x_z_ratio_k100, pts_y_z_ratio_k100 = pc_density_ellipsoid(pc_xyz, pc_cyflannKDTree_id, k=100)

    pc_cyflannKDTree_distance = None
    pc_cyflannKDTree_id = None

    DensityData = {'UTM-X': pc_xyz[:,0], 'UTM-Y': pc_xyz[:,1], 'UTM-Z': pc_xyz[:,2],
                   'Dens_sphere_k5': pc_dens_sphere_k5, 'Dens_sphere_k10': pc_dens_sphere_k10, 'Dens_sphere_k50': pc_dens_sphere_k50, 'Dens_sphere_k100': pc_dens_sphere_k100,
                   'Dens_ellipsoid_k5': pc_dens_ellipsoid_k5, 'Dens_ellipsoid_k10': pc_dens_ellipsoid_k10, 'Dens_ellipsoid_k50': pc_dens_ellipsoid_k50, 'Dens_ellipsoid_k100': pc_dens_ellipsoid_k100 }

    df_DensityData = pd.DataFrame(DensityData, columns= ['UTM-X', 'UTM-Y', 'UTM-Z', 'Dens_sphere_k5', 'Dens_sphere_k10', 'Dens_sphere_k50', 'Dens_sphere_k100', 'Dens_ellipsoid_k5', 'Dens_ellipsoid_k10', 'Dens_ellipsoid_k50', 'Dens_ellipsoid_k100'])
    #remove from memory
    DensityData = None

    XYZ_ratio = {'UTM-X': pc_xyz[:,0], 'UTM-Y': pc_xyz[:,1], 'UTM-Z': pc_xyz[:,2],
                   'pts_x_y_ratio_k10': pts_x_y_ratio_k10, 'pts_x_z_ratio_k10': pts_x_z_ratio_k10, 'pts_y_z_ratio_k10': pts_y_z_ratio_k10}

    df_DensityData = pd.DataFrame(DensityData, columns= ['UTM-X', 'UTM-Y', 'UTM-Z', 'Dens_sphere_k5', 'Dens_sphere_k10', 'Dens_sphere_k50', 'Dens_sphere_k100', 'Dens_ellipsoid_k5', 'Dens_ellipsoid_k10', 'Dens_ellipsoid_k50', 'Dens_ellipsoid_k100'])

    print(DensityData)
    DensityData.to_hdf(inps.hdf_filename, key='DensityData', complevel=9)

    if inps.write_to_LASFILES==True:
        ### Write to LAS/LAZ file (writing to LAZ file not yet supported by laspy, using work around with laszip)
        output_las_fn = '_density_sphere_k10.laz'
        output_las_fn = os.path.join(str(os.path.basename(inps.inlas).split('.')[:-1][0]) + output_las_fn)
        if os.path.exists(output_las_fn) == False:
            print('\tWriting density_sphere_k10 to LAZ file: %s... '%os.path.basename(output_las_fn), end='', flush=True)
            pts2write = pc_xyz
            mask = np.all(np.isnan(pts2write) | np.equal(pts2write, 0), axis=1)
            pts2write = pts2write[~mask]
            #normalize input and generate colors using colormap
            v = df_DensityData['Dens_sphere_k10'].values
            v = v[~mask]
            mask = None
            write_LAS(pts2write, v, output_las_fn[:-3]+'las', inps.inlas, cmap=mpl.cm.viridis)
            #To write LAZ file, using workaround with laszip and remove previous LAS file
            cmd = ['laszip', output_las_fn[:-3]+'las']
            logfile_fname = os.path.join(log_dir,  'laszip_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt')
            logfile_error_fname = os.path.join(log_dir, 'laszip_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt')
            with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
                subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
                subprocess_p.wait()
            if os.path.exists(output_las_fn[:-3]+'laz'):
                os.remove(output_las_fn[:-3]+'las')
            pts2write = None
            print('done.')

        ### Write to LAS/LAZ file (writing to LAZ file not yet supported by laspy, using work around with laszip)
        output_las_fn = '_density_ellipsoid_k10.laz'
        output_las_fn = os.path.join(str(os.path.basename(inps.inlas).split('.')[:-1][0]) + output_las_fn)
        if os.path.exists(output_las_fn) == False:
            print('\tWriting density_ellipsoid_k10 to LAZ file: %s... '%os.path.basename(output_las_fn), end='', flush=True)
            pts2write = pc_xyz
            mask = np.all(np.isnan(pts2write) | np.equal(pts2write, 0), axis=1)
            pts2write = pts2write[~mask]
            #normalize input and generate colors using colormap
            v = df_DensityData['Dens_ellipsoid_k10'].values
            v = v[~mask]
            mask = None
            write_LAS(pts2write, v, output_las_fn[:-3]+'las', inps.inlas, cmap=mpl.cm.viridis)
            #To write LAZ file, using workaround with laszip and remove previous LAS file
            cmd = ['laszip', output_las_fn[:-3]+'las']
            logfile_fname = os.path.join(log_dir,  'laszip_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt')
            logfile_error_fname = os.path.join(log_dir, 'laszip_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt')
            with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
                subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
                subprocess_p.wait()
            if os.path.exists(output_las_fn[:-3]+'laz'):
                os.remove(output_las_fn[:-3]+'las')
            pts2write = None
            print('done.')


        ### Write to LAS/LAZ file (writing to LAZ file not yet supported by laspy, using work around with laszip)
        output_las_fn = '_x_z_ratio_k10.laz'
        output_las_fn = os.path.join(str(os.path.basename(inps.inlas).split('.')[:-1][0]) + output_las_fn)
        if os.path.exists(output_las_fn) == False:
            print('\tWriting x_z_ratio_k10 to LAZ file: %s... '%os.path.basename(output_las_fn), end='', flush=True)
            pts2write = pc_xyz
            mask = np.all(np.isnan(pts2write) | np.equal(pts2write, 0), axis=1)
            pts2write = pts2write[~mask]
            #normalize input and generate colors using colormap
            v = pts_x_z_ratio_k10 #df_DensityData['Dens_ellipsoid_k10'].values
            v = v[~mask]
            mask = None
            write_LAS(pts2write, v, output_las_fn[:-3]+'las', inps.inlas, cmap=mpl.cm.viridis)
            #To write LAZ file, using workaround with laszip and remove previous LAS file
            cmd = ['laszip', output_las_fn[:-3]+'las']
            logfile_fname = os.path.join(log_dir,  'laszip_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt')
            logfile_error_fname = os.path.join(log_dir, 'laszip_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt')
            with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
                subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
                subprocess_p.wait()
            if os.path.exists(output_las_fn[:-3]+'laz'):
                os.remove(output_las_fn[:-3]+'las')
            pts2write = None
            print('done.')
