#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:36:08 2019

@author: bodo
"""

def write_LAS(pc_pc_xyz, v, output_las_fn, input_las_fn, cmap=mpl.cm.terrain, rescale='none'):
    from laspy.file import File

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


def pc_density_KDTree(xy, nn=51):
    n = xy.shape[0]
    p = np.zeros(n)
    #print('pc_density_cKDTree: Generating cKDTree')
    #print('pc_density_cKDTree: Find %d nearest neighbors in pcl_pc_xyzg_cKDTree:'%(nn, ))
    d,k = [pcl_xyg_kdtree_distance, pcl_xyg_kdtree_id] = pc_xyz_KDTree_tree.query(xy, k=nn, p=2)
    del k
    #euclidean distance p=2

    d = np.sqrt(d)
    dmin = d.min(axis = 0)
    dmin = dmin[-1]
    #print('PC density: min distance at k=%d: %0.3fm'%(nn,dmin))
    disk = np.pi * dmin * dmin
    for i in range(n):
        di = d[i]
        p[i] = len(di[di <= dmin]) / disk
    return p

def pc_density_scikit(xy, nn=51):
    n = xy.shape[0]
    p = np.zeros(n)
    #print('pc_density_cKDTree: Generating cKDTree')
    pc_xyz_KDTree_tree = sklearnKDTree(xy)
    #print('pc_density_cKDTree: Find %d nearest neighbors in pcl_pc_xyzg_cKDTree:'%(nn, ))
    d,k = [pcl_xyg_kdtree_distance, pcl_xyg_kdtree_id] = pc_xyz_KDTree_tree.query(xy, k=nn)
    del k
    #euclidean distance p=2

    d = np.sqrt(d)
    dmin = d.min(axis = 0)
    dmin = dmin[-1]
    #print('PC density: min distance at k=%d: %0.3fm'%(nn,dmin))
    disk = np.pi * dmin * dmin
    for i in range(n):
        di = d[i]
        p[i] = len(di[di <= dmin]) / disk
    return p


def pc_density_cKDTree(xy, nn=51):
    n = xy.shape[0]
    p = np.zeros(n)
    #print('pc_density_cKDTree: Generating cKDTree')
    pcl_pc_xyzg_cKDTree = spatial.cKDTree(xy, leafsize=32)
    #print('pc_density_cKDTree: Find %d nearest neighbors in pcl_pc_xyzg_cKDTree:'%(nn, ))
    d,k = [pcl_xyg_cKDTree_distance, pcl_xyg_cKDTree_id] = pcl_pc_xyzg_cKDTree.query(xy, k=nn, p=2, n_jobs=-1)
    del k
    #euclidean distance p=2

    d = np.sqrt(d)
    dmin = d.min(axis = 0)
    dmin = dmin[-1]
    #print('PC density: min distance at k=%d: %0.3fm'%(nn,dmin))
    disk = np.pi * dmin * dmin
    for i in range(n):
        di = d[i]
        p[i] = len(di[di <= dmin]) / disk
    return p

def pc_density_pyKDTree(xy, nn=51):
    n = xy.shape[0]
    p = np.zeros(n)
    #print('pc_density_pyKDTree: Generating pyKDTree')
    pcl_pc_xyzg_pyKDTree = pyKDTree(xy, leafsize=32)
    #print('pc_density_pyKDTree: Find %d nearest neighbors in pcl_pc_xyzg_pydtree:'%(nn, ))
    d,k = [pcl_xyg_pyKDTree_distance, pcl_xyg_pyKDTree_id] = pcl_pc_xyzg_pyKDTree.query(xy, k=nn)
    del k
    #euclidean distance p=2

    d = np.sqrt(d)
    dmin = d.min(axis = 0)
    dmin = dmin[-1]
    #print('PC density: min distance at k=%d: %0.3fm'%(nn,dmin))
    disk = np.pi * dmin * dmin
    for i in range(n):
        di = d[i]
        p[i] = len(di[di <= dmin]) / disk
    return p

def pc_density_pyflann(xy, nn=51):
    n = xy.shape[0]
    p = np.zeros(n)
    #print('pc_density_pyflann: Find %d nearest neighbors in pcl_pc_xyzg_cKDTree:'%(nn, ))

    pyflann.set_distance_type('euclidean')
    f = pyflann.FLANN()
    k, d = f.nn(xy, xy, 51, algorithm = 'kmeans', branching = 32, iterations = 7, checks = 16)
    del k

    d = np.sqrt(d)
    dmin = d.min(axis = 0)
    dmin = dmin[-1]
    #print('pc_density_pyflann: min distance at k=%d: %0.3fm'%(nn,dmin))
    disk = np.pi * dmin * dmin
    for i in range(n):
        di = d[i]
        p[i] = len(di[di <= dmin]) / disk
    return p
