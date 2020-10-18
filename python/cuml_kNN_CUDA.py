import cudf
import numpy as np
from cuml.datasets import make_blobs
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors
import argparse, numpy as np, pandas as pd, timeit, os, subprocess
import matplotlib.cm as cm

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

lasfname = 'Pozo_WestCanada_clg.las'
#inps.inlas = '/home/bodo/Dropbox/CampusGolm-Lidar/Golm_PC_Milan_MavicPro2_Inspire2/Golm_May06_2018_Milan_UTM33N_WGS84_6digit_cl_clip.las'
print('Loading input file: %s... '%lasfname, end='', flush=True)
pc_xyz = load_LAS(lasfname, dtype='float32')
pc_xyz[:,0] = pc_xyz[:,0] - np.nanmean(pc_xyz[:,0])
pc_xyz[:,1] = pc_xyz[:,1] - np.nanmean(pc_xyz[:,1])
pc_xyz[:,2] = pc_xyz[:,2] - np.nanmean(pc_xyz[:,2])
print('loaded %s points'%"{:,}".format(pc_xyz.shape[0]))

#generate Dataframe on CUDA device and copy data
gdf_float = cudf.DataFrame()
gdf_float['X'] = np.ascontiguousarray(pc_xyz[:,0])
gdf_float['Y'] = np.ascontiguousarray(pc_xyz[:,1])
gdf_float['Z'] = np.ascontiguousarray(pc_xyz[:,2])

print('n_samples = %d, n_dims = %d'%(pc_xyz.shape[0], pc_xyz.shape[1]))
print(gdf_float)

knn_cuml = cuNearestNeighbors()
knn_cuml.fit(gdf_float)
#find neighbors for every point
D_cuml3, I_cuml3 = knn_cuml.kneighbors(gdf_float, n_neighbors=3)
D_cuml10, I_cuml10 = knn_cuml.kneighbors(gdf_float, n_neighbors=10)

#Very fast and optimized
pc_xyz_cyflannKDTree_tree = pc_generate_cyflannKDTree(pc_xyz)
pc_cyflannKDTree_distance, pc_cyflannKDTree_id = pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=3)
pc_dens_sphere_k3 = pc_density_sphere(pc_cyflannKDTree_distance, k=3)

plt.scatter(pc_xyz[:,0], pc_xyz[:,1], pc_xyz[:,2], c=pc_dens_sphere_k3,
vmin=np.percentile(pc_dens_sphere_k3, 2), vmax=np.percentile(pc_dens_sphere_k3, 98))
plt.grid()
plt.xlabel('UTM-X [m]')
plt.ylabel('UTM-Y [m]')
plt.zlabel('Height [m]')

pc_cyflannKDTree_distance, pc_cyflannKDTree_id = pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=100)
pc_dens_sphere_k100 = pc_density_sphere(pc_cyflannKDTree_distance, k=100)

#Very slow
#copy back to host for sklearn approach
host_data = gdf_float.to_pandas()
knn_sk = skNearestNeighbors(algorithm="brute",
                            n_jobs=-1)
knn_sk.fit(host_data)
D_sk, I_sk = knn_sk.kneighbors(host_data, n_neighbors=3)
