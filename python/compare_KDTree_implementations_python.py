#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bodo Bookhagen, bodo.bookhagen@uni-potsdam.de
August 2019
updated Oct 2020

Example code that will compare the common KD-Tree implementation in
Python.
The code times the KD-Tree generation and then the query of
k=5,10,50,100,500,1000 nearest neighbors.

Running on
AMD Ryzen Threadripper 2970WX 24-Core Processor
Nvidia Titan RTX 24 GB
"""
import argparse, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pc_generate_KDTree(pc_xyz, leafsizei=10):
    try:
        from scipy import spatial
    except ImportError:
        raise pc_generate_KDTree("scipy not installed.")
    pc_xyz_KDTree_tree = spatial.KDTree(pc_xyz, leafsize=leafsizei)
    return pc_xyz_KDTree_tree

def pc_query_KDTree(pc_xyz_KDTree_tree, pc_xyz, k=10):
    pc_kdtree_distance, pc_kdtree_id = pc_xyz_KDTree_tree.query(pc_xyz, k=k)
    return pc_kdtree_distance, pc_kdtree_id

def pc_generate_sklearnKDTree(pc_xyz, leafsizei=10):
    #conda install scikit-learn
    try:
        from sklearn.neighbors import KDTree as sklearnKDTree
    except ImportError:
        raise pc_generate_sklearnKDTree("sklearn not installed.")
    pc_xyz_sklearnKDTree_tree = sklearnKDTree(pc_xyz, leaf_size=leafsizei)
    return pc_xyz_sklearnKDTree_tree

def pc_query_sklearnKDTree(pc_xyz_sklearnKDTree_tree, pc_xyz, k=10):
    pc_sklearnKDTree_distance, pc_sklearnKDTree_id = pc_xyz_sklearnKDTree_tree.query(pc_xyz, k=k, dualtree=True)
    return pc_sklearnKDTree_distance, pc_sklearnKDTree_id

def pc_generate_cKDTree(pc_xyz, leafsizei=10):
    try:
        from scipy import spatial
    except ImportError:
        raise pc_generate_cKDTree("scipy not installed.")
    pc_xyz_cKDTree_tree = spatial.cKDTree(pc_xyz, leafsize=leafsizei)
    return pc_xyz_cKDTree_tree

def pc_query_cKDTree(pc_xyz_cKDTree_tree, pc_xyz, k=10):
    pc_cKDTree_distance, pc_cKDTree_id = pc_xyz_cKDTree_tree.query(pc_xyz, k=k, n_jobs=-1)
    return pc_cKDTree_distance, pc_cKDTree_id

def pc_generate_pyKDTree(pc_xyz, leafsizei=10):
    try:
        from pykdtree.kdtree import KDTree as pyKDTree
    except ImportError:
        raise pc_generate_pyKDTree("pykdtree not installed.")
    pc_xyz_pyKDTree_tree = pyKDTree(pc_xyz, leafsize=leafsizei)
    return pc_xyz_pyKDTree_tree

def pc_query_pyKDTree(pc_xyz_pyKDTree_tree, pc_xyz, k=10):
    pc_pyKDTree_distance, pc_pyKDTree_id = pc_xyz_pyKDTree_tree.query(pc_xyz, k=k)
    return pc_pyKDTree_distance, pc_pyKDTree_id

def pc_generate_pyflannKDTree(pc_xyz):
    #conda install -y -c conda-forge pyflann
    try:
        import pyflann
    except ImportError:
        raise pc_generate_pyflannKDTree("pyflann not installed.")
    pyflann.set_distance_type('euclidean')
    pc_xyz_pyflannKDTree_tree = pyflann.FLANN()
    pc_xyz_pyflannKDTree_tree.build_index(pc_xyz, algorithm='kdtree_single', trees=8)
    return pc_xyz_pyflannKDTree_tree

def pc_query_pyflannKDTree(pc_xyz_pyflannKDTree_tree, pc_xyz, k=10):
    pc_pyflannKDTree_id, pc_pyflannKDTree_distance = pc_xyz_pyflannKDTree_tree.nn_index(pc_xyz, k)
    return  pc_pyflannKDTree_distance, pc_pyflannKDTree_id

def pc_generate_cyflannKDTree(pc_xyz):
    #conda install -y -c conda-forge cyflann
    try:
        import cyflann
    except ImportError:
        raise pc_generate_cyflannKDTree("pyflann not installed.")
    cyflann.set_distance_type('euclidean')
    pc_xyz_cyflannKDTree_tree = cyflann.FLANNIndex()
    pc_xyz_cyflannKDTree_tree.build_index(pc_xyz, algorithm='kdtree_single', trees=8)
    return pc_xyz_cyflannKDTree_tree

def pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=10):
    pc_cyflannKDTree_id, pc_cyflannKDTree_distance = pc_xyz_cyflannKDTree_tree.nn_index(pc_xyz, k)
    return pc_cyflannKDTree_distance, pc_cyflannKDTree_id

def pc_generate_cudf_NN(pc_xyz):
    try:
        import cudf
    except ImportError:
        raise pc_generate_cudf_NN("cudf not installed.")
    try:
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    except ImportError:
        raise pc_generate_cudf_NN("cuml not installed.")
    gdf_float = cudf.DataFrame()
    gdf_float['X'] = np.ascontiguousarray(pc_xyz[:,0])
    gdf_float['Y'] = np.ascontiguousarray(pc_xyz[:,1])
    gdf_float['Z'] = np.ascontiguousarray(pc_xyz[:,2])
    #one-liner:
    #gdf_float = pd.DataFrame({'X':pc_xyz[:,0], 'Y':pc_xyz[:,1], 'Z':pc_xyz[:,2],})

    knn_cuml = cuNearestNeighbors()
    knn_cuml.fit(gdf_float)
    return knn_cuml

def pc_query_cudf_NN(knn_cuml, gdf_float, k=3):
    D_cuml, I_cuml = knn_cuml.kneighbors(gdf_float, n_neighbors=k)
    return knn_cuml_D, knncuml_I

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

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
    parser.add_argument('--nr_of_repetitions', type=int, default=5,  help='Number of repititions')
    parser.add_argument('--hdf_filename', type=str, default='Pozo_WestCanada_clg_kdresults_5rep.hdf',  help='Output HDF file containting results from iterations.')
    parser.add_argument('--csv_filename', type=str, default='Pozo_WestCanada_clg_kdresults_5rep.csv',  help='Output CSV file containting results from iterations.')
    parser.add_argument('--nr_of_repetitions_generate', type=int, default=1,  help='Number of repititions used for generating index. Set to 1 to avoid caching effects.')
    return parser.parse_args()

def pandas_df_to_markdown_table(df):
    # Dependent upon ipython
    # shamelessly stolen from https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    #display(Markdown(df_formatted.to_csv(sep="|", index=False)))
    return Markdown(df_formatted.to_csv(sep="|", index=False))
#     return df_formatted

def df_to_markdown(df, y_index=False):
    from tabulate import tabulate
    blob = tabulate(df, headers='keys', tablefmt='pipe')
    if not y_index:
        # Remove the index with some creative splicing and iteration
        return '\n'.join(['| {}'.format(row.split('|', 2)[-1]) for row in blob.split('\n')])
    return blob

### Main
if __name__ == '__main__':
    inps = cmdLineParser()
# Testing
    inps = argparse.ArgumentParser(description='Compare KDTree algorithms for lidar or SfM PointClouds (PC). Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de), Aug 2019.')
    inps.inlas = 'Pozo_WestCanada_clg.laz'
    inps.nr_of_repetitions = 3
    inps.nr_of_repetitions_generate = 1

    #kant: 3.9 GHz
    inps.cpuname= 'AMDRyzen_3900X_12cores'
    inps.hdf_filename = 'Pozo_WestCanada_%s_rep3.hdf'%(inps.cpuname)
    inps.csv_filename = 'Pozo_WestCanada_%s_rep3.hdf'%(inps.cpuname)

    #macon: 2.9 GHz
    inps.cpuname= 'AMDRyzen_2970WX_24cores'
    inps.hdf_filename = 'Pozo_WestCanada_%s_rep3.hdf'%(inps.cpuname)
    inps.csv_filename = 'Pozo_WestCanada_%s_rep3.csv'%(inps.cpuname)

    #aconcagua: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz, 2x20 cores, 2x40 threads
    inps.cpuname= 'Xeon_Gold6230_40cores'
    inps.hdf_filename = 'Pozo_WestCanada_%s_rep3.hdf'%(inps.cpuname)
    inps.csv_filename = 'Pozo_WestCanada_%s_rep3.hdf'%(inps.cpuname)

    print('Loading input file: %s... '%inps.inlas, end='', flush=True)
    pc_xyz = load_LAS(inps.inlas, dtype='float32')
    print('loaded %s points'%"{:,}".format(pc_xyz.shape[0]))

    #Run standard KDTree (python implementation from scipy)
    ## Not RUNNING KDTREE querying, because it is too slow.
    leafrange = range(8,40,2)
    pc_generate_KDTree_time = np.empty( (len(leafrange), 1) )
    pc_generate_KDTree_time[:] = np.nan
    pc_query_KDTree_k5_time = np.empty( (len(leafrange), 1) )
    pc_query_KDTree_k5_time[:] = np.nan
    pc_query_KDTree_k10_time = np.empty( (len(leafrange), 1) )
    pc_query_KDTree_k10_time[:] = np.nan
    pc_query_KDTree_k50_time = np.empty( (len(leafrange), 1) )
    pc_query_KDTree_k50_time[:] = np.nan
    pc_query_KDTree_k100_time = np.empty( (len(leafrange), 1) )
    pc_query_KDTree_k100_time[:] = np.nan
    pc_query_KDTree_k500_time = np.empty( (len(leafrange), 1) )
    pc_query_KDTree_k500_time[:] = np.nan
    pc_query_KDTree_k1000_time = np.empty( (len(leafrange), 1) )
    pc_query_KDTree_k1000_time[:] = np.nan
    pc_query_KDTree_k5_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_KDTree_k5_stats[:] = np.nan
    pc_query_KDTree_k10_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_KDTree_k10_stats[:] = np.nan

    leaf_counter = 0
    for leafsizei in leafrange:
        print('\n\tGenerating KDTree with leafsize = %d ... (%dx) '%(leafsizei, inps.nr_of_repetitions_generate), end='', flush=True)
        wrapped = wrapper(pc_generate_KDTree, pc_xyz, leafsizei)
        pc_generate_KDTree_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions_generate)
        pc_xyz_KDTree_tree = pc_generate_KDTree(pc_xyz, leafsizei=leafsizei)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions_generate, pc_generate_KDTree_time[leaf_counter]/inps.nr_of_repetitions, pc_generate_KDTree_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying KDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=5)
        pc_query_KDTree_k5_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k5_time[leaf_counter]/inps.nr_of_repetitions, pc_query_KDTree_k5_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_KDTree_distance, pc_KDTree_id = pc_query_KDTree(pc_xyz_KDTree_tree, pc_xyz, k=5)
        pc_query_KDTree_k5_stats[leaf_counter, :, 0] = np.mean(pc_KDTree_distance, axis=1)
        pc_query_KDTree_k5_stats[leaf_counter, :, 1] = np.std(pc_KDTree_distance, axis=1)
        pc_query_KDTree_k5_stats[leaf_counter, :, 2] = np.percentile(pc_KDTree_distance, [25], axis=1)
        pc_query_KDTree_k5_stats[leaf_counter, :, 3] = np.percentile(pc_KDTree_distance, [50], axis=1)
        pc_query_KDTree_k5_stats[leaf_counter, :, 4] = np.percentile(pc_KDTree_distance, [75], axis=1)
        pc_KDTree_distance = None
        pc_KDTree_id = None

        # print('\tQuerying KDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        # wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=10)
        # pc_query_KDTree_k10_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        # print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k10_time/inps.nr_of_repetitions, pc_query_KDTree_k10_time/inps.nr_of_repetitions/60))
        # pc_KDTree_distance, pc_KDTree_id = pc_query_KDTree(pc_xyz_KDTree_tree, pc_xyz, k=10)
        # pc_query_KDTree_k10_stats[leaf_counter, :, 0] = np.mean(pc_KDTree_distance, axis=1)
        # pc_query_KDTree_k10_stats[leaf_counter, :, 1] = np.std(pc_KDTree_distance, axis=1)
        # pc_query_KDTree_k10_stats[leaf_counter, :, 2] = np.percentile(pc_KDTree_distance, [25], axis=1)
        # pc_query_KDTree_k10_stats[leaf_counter, :, 3] = np.percentile(pc_KDTree_distance, [50], axis=1)
        # pc_query_KDTree_k10_stats[leaf_counter, :, 4] = np.percentile(pc_KDTree_distance, [75], axis=1)
        # pc_KDTree_distance = None
        # pc_KDTree_id = None

        leaf_counter += 1

#
#    print('\tQuerying KDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=50)
#    pc_query_KDTree_k50_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k50_time/inps.nr_of_repetitions, pc_query_KDTree_k50_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying KDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=100)
#    pc_query_KDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k100_time/inps.nr_of_repetitions, pc_query_KDTree_k100_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying KDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=500)
#    pc_query_KDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k500_time/inps.nr_of_repetitions, pc_query_KDTree_k500_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying KDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=1000)
#    pc_query_KDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k1000_time/inps.nr_of_repetitions, pc_query_KDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_KDTree_tree = None

    #Run sklearnKDTree
    pc_generate_sklearnKDTree_time = np.empty( (len(leafrange), 1) )
    pc_generate_sklearnKDTree_time[:] = np.nan
    pc_query_sklearnKDTree_k5_time = np.empty( (len(leafrange), 1) )
    pc_query_sklearnKDTree_k5_time[:] = np.nan
    pc_query_sklearnKDTree_k10_time = np.empty( (len(leafrange), 1) )
    pc_query_sklearnKDTree_k10_time[:] = np.nan
    pc_query_sklearnKDTree_k50_time = np.empty( (len(leafrange), 1) )
    pc_query_sklearnKDTree_k50_time[:] = np.nan
    pc_query_sklearnKDTree_k100_time = np.empty( (len(leafrange), 1) )
    pc_query_sklearnKDTree_k100_time[:] = np.nan
    pc_query_sklearnKDTree_k500_time = np.empty( (len(leafrange), 1) )
    pc_query_sklearnKDTree_k500_time[:] = np.nan
    pc_query_sklearnKDTree_k1000_time = np.empty( (len(leafrange), 1) )
    pc_query_sklearnKDTree_k1000_time[:] = np.nan
    pc_query_sklearnKDTree_k5_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_sklearnKDTree_k5_stats[:] = np.nan
    pc_query_sklearnKDTree_k10_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_sklearnKDTree_k10_stats[:] = np.nan

    leaf_counter = 0
    for leafsizei in leafrange:
        print('\n\tGenerating sklearnKDTree... with leafsize = %d (%dx) '%(leafsizei, inps.nr_of_repetitions_generate), end='', flush=True)
        wrapped = wrapper(pc_generate_sklearnKDTree, pc_xyz, leafsizei)
        pc_generate_sklearnKDTree_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions_generate)
        pc_xyz_sklearnKDTree_tree = pc_generate_sklearnKDTree(pc_xyz, leafsizei=leafsizei)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions_generate, pc_generate_sklearnKDTree_time[leaf_counter]/inps.nr_of_repetitions, pc_generate_sklearnKDTree_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying sklearnKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=5)
        pc_query_sklearnKDTree_k5_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time[leaf_counter]/inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_sklearnKDTree_distance, pc_sklearnKDTree_id = pc_query_sklearnKDTree(pc_xyz_sklearnKDTree_tree, pc_xyz, k=5)
        pc_query_sklearnKDTree_k5_stats[leaf_counter, :, 0] = np.mean(pc_sklearnKDTree_distance, axis=1)
        pc_query_sklearnKDTree_k5_stats[leaf_counter, :, 1] = np.std(pc_sklearnKDTree_distance, axis=1)
        pc_query_sklearnKDTree_k5_stats[leaf_counter, :, 2] = np.percentile(pc_sklearnKDTree_distance, [25], axis=1)
        pc_query_sklearnKDTree_k5_stats[leaf_counter, :, 3] = np.percentile(pc_sklearnKDTree_distance, [50], axis=1)
        pc_query_sklearnKDTree_k5_stats[leaf_counter, :, 4] = np.percentile(pc_sklearnKDTree_distance, [75], axis=1)
        pc_sklearnKDTree_distance = None
        pc_sklearnKDTree_id = None

        # print('\tQuerying sklearnKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        # wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=10)
        # pc_query_sklearnKDTree_k10_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        # print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time[leaf_counter]/inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time[leaf_counter]/inps.nr_of_repetitions/60))
        # pc_sklearnKDTree_distance, pc_sklearnKDTree_id = pc_query_sklearnKDTree(pc_xyz_sklearnKDTree_tree, pc_xyz, k=10)
        # pc_query_sklearnKDTree_k10_stats[leaf_counter, :, 0] = np.mean(pc_sklearnKDTree_distance, axis=1)
        # pc_query_sklearnKDTree_k10_stats[leaf_counter, :, 1] = np.std(pc_sklearnKDTree_distance, axis=1)
        # pc_query_sklearnKDTree_k10_stats[leaf_counter, :, 2] = np.percentile(pc_sklearnKDTree_distance, [25], axis=1)
        # pc_query_sklearnKDTree_k10_stats[leaf_counter, :, 3] = np.percentile(pc_sklearnKDTree_distance, [50], axis=1)
        # pc_query_sklearnKDTree_k10_stats[leaf_counter, :, 4] = np.percentile(pc_sklearnKDTree_distance, [75], axis=1)
        # pc_sklearnKDTree_distance = None
        # pc_sklearnKDTree_id = None

        # print('\tQuerying sklearnKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        # wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=50)
        # pc_query_sklearnKDTree_k50_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        # print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time[leaf_counter]/inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time[leaf_counter]/inps.nr_of_repetitions/60))
        #
        # print('\tQuerying sklearnKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        # wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=100)
        # pc_query_sklearnKDTree_k100_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        # print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k100_time[leaf_counter]/inps.nr_of_repetitions, pc_query_sklearnKDTree_k100_time[leaf_counter]/inps.nr_of_repetitions/60))
        #
        # print('\tQuerying sklearnKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        # wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=500)
        # pc_query_sklearnKDTree_k500_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        # print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k500_time[leaf_counter]/inps.nr_of_repetitions, pc_query_sklearnKDTree_k500_time[leaf_counter]/inps.nr_of_repetitions/60))
        #
        # print('\tQuerying sklearnKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        # wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=1000)
        # pc_query_sklearnKDTree_k1000_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        # print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k1000_time[leaf_counter]/inps.nr_of_repetitions, pc_query_sklearnKDTree_k1000_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_xyz_sklearnKDTree_tree = None
        leaf_counter += 1

    # plt.clf()
    # plt.plot(leafrange, pc_query_sklearnKDTree_k5_time/inps.nr_of_repetitions, 'kx-', label='query sklearnKDTree k=5')
    # plt.plot(leafrange, (pc_query_sklearnKDTree_k5_time)/inps.nr_of_repetitions+pc_generate_sklearnKDTree_time/inps.nr_of_repetitions_generate, 'x-', c='gray', label='create + query sklearnKDTree k=5')
    # plt.title('sklearnKDTree (single core): generation and query (AMD3900X: 12 cores)')
    # plt.grid()
    # plt.xlabel('leafsize')
    # plt.ylabel('Query time (s)')
    # plt.xlim([8,30])
    # plt.xticks(np.arange(8,30,step=2))
    # #plt.ylim([0,2])
    # plt.legend()
    # plt.savefig('figs/pc_sklearnKDTree_AMD3900X_12cores.png', dpi=300, orientation='landscape')

    #Save outputs to compressed HDF file
    # need to combine into dataframe. Dataframe should contain x, y, z coordinate and then the statistics for each point and for each leafsize
    #pc_xyz
    #pc_sklearnKDTree_stats_df = pd.DataFrame({'pc_query_sklearnKDTree_k5_stats': pc_query_sklearnKDTree_k5_stats})
    # pc_sklearnKDTree_time_df = pd.DataFrame({'pc_query_sklearnKDTree_k5_time': pc_query_sklearnKDTree_k5_time,
    #     'pc_generate_sklearnKDTree_time': pc_generate_sklearnKDTree_time})
    # pc_sklearnKDTree_df.to_hdf('pc_sklearnKDTree_stats_df.hdf', key='pc_sklearnKDTree_stats_df', complevel=9)


    #Run cKDTree (cython implementation from scipy)
    pc_generate_cKDTree_time = np.empty( (len(leafrange), 1) )
    pc_generate_cKDTree_time[:] = np.nan
    pc_query_cKDTree_k5_time = np.empty( (len(leafrange), 1) )
    pc_query_cKDTree_k5_time[:] = np.nan
    pc_query_cKDTree_k10_time = np.empty( (len(leafrange), 1) )
    pc_query_cKDTree_k10_time[:] = np.nan
    pc_query_cKDTree_k50_time = np.empty( (len(leafrange), 1) )
    pc_query_cKDTree_k50_time[:] = np.nan
    pc_query_cKDTree_k100_time = np.empty( (len(leafrange), 1) )
    pc_query_cKDTree_k100_time[:] = np.nan
    pc_query_cKDTree_k500_time = np.empty( (len(leafrange), 1) )
    pc_query_cKDTree_k500_time[:] = np.nan
    pc_query_cKDTree_k1000_time = np.empty( (len(leafrange), 1) )
    pc_query_cKDTree_k1000_time[:] = np.nan
    pc_query_cKDTree_k5_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_cKDTree_k5_stats[:] = np.nan
    pc_query_cKDTree_k10_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_cKDTree_k10_stats[:] = np.nan

    leaf_counter = 0
    for leafsizei in leafrange:
        print('\n\tGenerating cKDTree with leafsize = %d ... (%dx) '%(leafsizei, inps.nr_of_repetitions_generate), end='', flush=True)
        wrapped = wrapper(pc_generate_cKDTree, pc_xyz, leafsizei)
        pc_generate_cKDTree_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions_generate)
        pc_xyz_cKDTree_tree = pc_generate_cKDTree(pc_xyz, leafsizei=leafsizei)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions_generate, pc_generate_cKDTree_time[leaf_counter]/inps.nr_of_repetitions, pc_generate_cKDTree_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying cKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=5)
        pc_query_cKDTree_k5_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k5_time[leaf_counter]/inps.nr_of_repetitions, pc_query_cKDTree_k5_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_cKDTree_distance, pc_cKDTree_id = pc_query_cKDTree(pc_xyz_cKDTree_tree, pc_xyz, k=5)
        pc_query_cKDTree_k5_stats[leaf_counter, :, 0] = np.mean(pc_cKDTree_distance, axis=1)
        pc_query_cKDTree_k5_stats[leaf_counter, :, 1] = np.std(pc_cKDTree_distance, axis=1)
        pc_query_cKDTree_k5_stats[leaf_counter, :, 2] = np.percentile(pc_cKDTree_distance, [25], axis=1)
        pc_query_cKDTree_k5_stats[leaf_counter, :, 3] = np.percentile(pc_cKDTree_distance, [50], axis=1)
        pc_query_cKDTree_k5_stats[leaf_counter, :, 4] = np.percentile(pc_cKDTree_distance, [75], axis=1)
        pc_cKDTree_distance = None
        pc_cKDTree_id = None

        print('\tQuerying cKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=10)
        pc_query_cKDTree_k10_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k10_time[leaf_counter]/inps.nr_of_repetitions, pc_query_cKDTree_k10_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_cKDTree_distance, pc_cKDTree_id = pc_query_cKDTree(pc_xyz_cKDTree_tree, pc_xyz, k=10)
        pc_query_cKDTree_k10_stats[leaf_counter, :, 0] = np.mean(pc_cKDTree_distance, axis=1)
        pc_query_cKDTree_k10_stats[leaf_counter, :, 1] = np.std(pc_cKDTree_distance, axis=1)
        pc_query_cKDTree_k10_stats[leaf_counter, :, 2] = np.percentile(pc_cKDTree_distance, [25], axis=1)
        pc_query_cKDTree_k10_stats[leaf_counter, :, 3] = np.percentile(pc_cKDTree_distance, [50], axis=1)
        pc_query_cKDTree_k10_stats[leaf_counter, :, 4] = np.percentile(pc_cKDTree_distance, [75], axis=1)
        pc_cKDTree_distance = None
        pc_cKDTree_id = None

        print('\tQuerying cKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=50)
        pc_query_cKDTree_k50_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k50_time[leaf_counter]/inps.nr_of_repetitions, pc_query_cKDTree_k50_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying cKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=100)
        pc_query_cKDTree_k100_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k100_time[leaf_counter]/inps.nr_of_repetitions, pc_query_cKDTree_k100_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying cKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=500)
        pc_query_cKDTree_k500_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k500_time[leaf_counter]/inps.nr_of_repetitions, pc_query_cKDTree_k500_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying cKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=1000)
        pc_query_cKDTree_k1000_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k1000_time[leaf_counter]/inps.nr_of_repetitions, pc_query_cKDTree_k1000_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_xyz_cKDTree_tree = None
        leaf_counter += 1

    pc_cKDTree_time_df = pd.DataFrame({'index': range(len(leafrange)), 'leafsize': leafrange,
        'pc_query_cKDTree_k5_time': pc_query_cKDTree_k5_time.ravel(),
        'pc_query_cKDTree_k10_time': pc_query_cKDTree_k10_time.ravel(),
        'pc_query_cKDTree_k50_time': pc_query_cKDTree_k50_time.ravel(),
        'pc_query_cKDTree_k100_time': pc_query_cKDTree_k100_time.ravel(),
        'pc_query_cKDTree_k500_time': pc_query_cKDTree_k500_time.ravel(),
        'pc_query_cKDTree_k1000_time': pc_query_cKDTree_k1000_time.ravel(),
        'pc_generate_cKDTree_time': pc_generate_cKDTree_time.ravel()})
    pc_cKDTree_time_df.to_hdf('pc_cKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cKDTree_time_df', complevel=9)

    #For simplicity, save array to npy
    np.save('pc_query_cKDTree_k5_stats_%s.npy'%(inps.cpuname),pc_query_cKDTree_k5_stats)
    np.save('pc_query_cKDTree_k10_stats_%s.npy'%(inps.cpuname),pc_query_cKDTree_k10_stats)

    #Run pyKDTree (fast implementation using c and libANN)
    pc_generate_pyKDTree_time = np.empty( (len(leafrange), 1) )
    pc_generate_pyKDTree_time[:] = np.nan
    pc_query_pyKDTree_k5_time = np.empty( (len(leafrange), 1) )
    pc_query_pyKDTree_k5_time[:] = np.nan
    pc_query_pyKDTree_k10_time = np.empty( (len(leafrange), 1) )
    pc_query_pyKDTree_k10_time[:] = np.nan
    pc_query_pyKDTree_k50_time = np.empty( (len(leafrange), 1) )
    pc_query_pyKDTree_k50_time[:] = np.nan
    pc_query_pyKDTree_k100_time = np.empty( (len(leafrange), 1) )
    pc_query_pyKDTree_k100_time[:] = np.nan
    pc_query_pyKDTree_k500_time = np.empty( (len(leafrange), 1) )
    pc_query_pyKDTree_k500_time[:] = np.nan
    pc_query_pyKDTree_k1000_time = np.empty( (len(leafrange), 1) )
    pc_query_pyKDTree_k1000_time[:] = np.nan
    pc_query_pyKDTree_k5_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_pyKDTree_k5_stats[:] = np.nan
    pc_query_pyKDTree_k10_stats = np.empty( (len(leafrange), len(pc_xyz), 5) )
    pc_query_pyKDTree_k10_stats[:] = np.nan

    leaf_counter = 0
    for leafsizei in leafrange:
        print('\n\tGenerating pyKDTree... with leafsize = %d (%dx) '%(leafsizei, inps.nr_of_repetitions_generate), end='', flush=True)
        wrapped = wrapper(pc_generate_pyKDTree, pc_xyz, leafsizei)
        pc_generate_pyKDTree_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions_generate)
        pc_xyz_pyKDTree_tree = pc_generate_pyKDTree(pc_xyz, leafsizei=leafsizei)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions_generate, pc_generate_pyKDTree_time[leaf_counter]/inps.nr_of_repetitions, pc_generate_pyKDTree_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying pyKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=5)
        pc_query_pyKDTree_k5_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k5_time[leaf_counter]/inps.nr_of_repetitions, pc_query_pyKDTree_k5_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_pyKDTree_distance, pc_pyKDTree_id = pc_query_pyKDTree(pc_xyz_pyKDTree_tree, pc_xyz, k=5)
        pc_query_pyKDTree_k5_stats[leaf_counter, :, 0] = np.mean(pc_pyKDTree_distance, axis=1)
        pc_query_pyKDTree_k5_stats[leaf_counter, :, 1] = np.std(pc_pyKDTree_distance, axis=1)
        pc_query_pyKDTree_k5_stats[leaf_counter, :, 2] = np.percentile(pc_pyKDTree_distance, [25], axis=1)
        pc_query_pyKDTree_k5_stats[leaf_counter, :, 3] = np.percentile(pc_pyKDTree_distance, [50], axis=1)
        pc_query_pyKDTree_k5_stats[leaf_counter, :, 4] = np.percentile(pc_pyKDTree_distance, [75], axis=1)
        pc_pyKDTree_distance = None
        pc_pyKDTree_id = None

        print('\tQuerying pyKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=10)
        pc_query_pyKDTree_k10_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k10_time[leaf_counter]/inps.nr_of_repetitions, pc_query_pyKDTree_k10_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_pyKDTree_distance, pc_pyKDTree_id = pc_query_pyKDTree(pc_xyz_pyKDTree_tree, pc_xyz, k=10)
        pc_query_pyKDTree_k10_stats[leaf_counter, :, 0] = np.mean(pc_pyKDTree_distance, axis=1)
        pc_query_pyKDTree_k10_stats[leaf_counter, :, 1] = np.std(pc_pyKDTree_distance, axis=1)
        pc_query_pyKDTree_k10_stats[leaf_counter, :, 2] = np.percentile(pc_pyKDTree_distance, [25], axis=1)
        pc_query_pyKDTree_k10_stats[leaf_counter, :, 3] = np.percentile(pc_pyKDTree_distance, [50], axis=1)
        pc_query_pyKDTree_k10_stats[leaf_counter, :, 4] = np.percentile(pc_pyKDTree_distance, [75], axis=1)
        pc_pyKDTree_distance = None
        pc_pyKDTree_id = None

        print('\tQuerying pyKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=50)
        pc_query_pyKDTree_k50_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k50_time[leaf_counter]/inps.nr_of_repetitions, pc_query_pyKDTree_k50_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying pyKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=100)
        pc_query_pyKDTree_k100_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k100_time[leaf_counter]/inps.nr_of_repetitions, pc_query_pyKDTree_k100_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying pyKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=500)
        pc_query_pyKDTree_k500_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k500_time[leaf_counter]/inps.nr_of_repetitions, pc_query_pyKDTree_k500_time[leaf_counter]/inps.nr_of_repetitions/60))

        print('\tQuerying pyKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
        wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=1000)
        pc_query_pyKDTree_k1000_time[leaf_counter] = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
        print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k1000_time[leaf_counter]/inps.nr_of_repetitions, pc_query_pyKDTree_k1000_time[leaf_counter]/inps.nr_of_repetitions/60))
        pc_xyz_pyKDTree_tree = None
        leaf_counter += 1

    pc_pyKDTree_time_df = pd.DataFrame({'index': range(len(leafrange)), 'leafsize': leafrange,
        'pc_query_pyKDTree_k5_time': pc_query_pyKDTree_k5_time.ravel(),
        'pc_query_pyKDTree_k10_time': pc_query_pyKDTree_k10_time.ravel(),
        'pc_query_pyKDTree_k50_time': pc_query_pyKDTree_k50_time.ravel(),
        'pc_query_pyKDTree_k100_time': pc_query_pyKDTree_k100_time.ravel(),
        'pc_query_pyKDTree_k500_time': pc_query_pyKDTree_k500_time.ravel(),
        'pc_query_pyKDTree_k1000_time': pc_query_pyKDTree_k1000_time.ravel(),
        'pc_generate_pyKDTree_time': pc_generate_pyKDTree_time.ravel()})
    pc_pyKDTree_time_df.to_hdf('pc_pyKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyKDTree_time_df', complevel=9)
    np.save('pc_query_pyKDTree_k5_stats_%s.npy'%(inps.cpuname),pc_query_pyKDTree_k5_stats)
    np.save('pc_query_pyKDTree_k10_stats_%s.npy'%(inps.cpuname),pc_query_pyKDTree_k10_stats)

    #Run pyflannKDTree
    print('\n\tGenerating pyflannKDTree... (%dx) '%(inps.nr_of_repetitions_generate), end='', flush=True)
    wrapped = wrapper(pc_generate_pyflannKDTree, pc_xyz)
    pc_generate_pyflannKDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions_generate)
    pc_xyz_pyflannKDTree_tree = pc_generate_pyflannKDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions_generate, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyflannKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=5)
    pc_query_pyflannKDTree_k5_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k5_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k5_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyflannKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=10)
    pc_query_pyflannKDTree_k10_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k10_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k10_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyflannKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=50)
    pc_query_pyflannKDTree_k50_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k50_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k50_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyflannKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=100)
    pc_query_pyflannKDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k100_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k100_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyflannKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=500)
    pc_query_pyflannKDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k500_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k500_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyflannKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=1000)
    pc_query_pyflannKDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k1000_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_pyflannKDTree_tree = None

    pc_pyflannKDTree_time_df = pd.DataFrame({'index': [0],
        'pc_query_pyflannKDTree_k5_time': pc_query_pyflannKDTree_k5_time,
        'pc_query_pyflannKDTree_k10_time': pc_query_pyflannKDTree_k10_time,
        'pc_query_pyflannKDTree_k50_time': pc_query_pyflannKDTree_k50_time,
        'pc_query_pyflannKDTree_k100_time': pc_query_pyflannKDTree_k100_time,
        'pc_query_pyflannKDTree_k500_time': pc_query_pyflannKDTree_k500_time,
        'pc_query_pyflannKDTree_k1000_time': pc_query_pyflannKDTree_k1000_time,
        'pc_generate_pyflannKDTree_time': pc_generate_pyflannKDTree_time})
    pc_pyflannKDTree_time_df.to_hdf('pc_pyflannKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyflannKDTree_time_df', complevel=9)

    #Run cyflannKDTree
    print('\n\tGenerating cyflannKDTree... (%dx) '%(inps.nr_of_repetitions_generate), end='', flush=True)
    wrapped = wrapper(pc_generate_cyflannKDTree, pc_xyz)
    pc_generate_cyflannKDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions_generate)
    pc_xyz_cyflannKDTree_tree = pc_generate_cyflannKDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions_generate, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions/60))

    print('\tQuerying cyflannKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=5)
    pc_query_cyflannKDTree_k5_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k5_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k5_time/inps.nr_of_repetitions/60))

    print('\tQuerying cyflannKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=10)
    pc_query_cyflannKDTree_k10_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k10_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k10_time/inps.nr_of_repetitions/60))

    print('\tQuerying cyflannKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=50)
    pc_query_cyflannKDTree_k50_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k50_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k50_time/inps.nr_of_repetitions/60))

    print('\tQuerying cyflannKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=100)
    pc_query_cyflannKDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k100_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k100_time/inps.nr_of_repetitions/60))

    print('\tQuerying cyflannKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=500)
    pc_query_cyflannKDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k500_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k500_time/inps.nr_of_repetitions/60))

    print('\tQuerying cyflannKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=1000)
    pc_query_cyflannKDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k1000_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_cyflannKDTree_tree = None

    pc_cyflannKDTree_time_df = pd.DataFrame({'index': [0], 'pc_query_cyflannKDTree_k5_time': pc_query_cyflannKDTree_k5_time,
        'pc_query_cyflannKDTree_k10_time': pc_query_cyflannKDTree_k10_time,
        'pc_query_cyflannKDTree_k50_time': pc_query_cyflannKDTree_k50_time,
        'pc_query_cyflannKDTree_k100_time': pc_query_cyflannKDTree_k100_time,
        'pc_query_cyflannKDTree_k500_time': pc_query_cyflannKDTree_k500_time,
        'pc_query_cyflannKDTree_k1000_time': pc_query_cyflannKDTree_k1000_time,
        'pc_generate_cyflannKDTree_time': pc_generate_cyflannKDTree_time})
    pc_cyflannKDTree_time_df.to_hdf('pc_cyflannKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cyflannKDTree_time_df', complevel=9)

    TimeTable_besttimes = {'Algorithm': ['KDTree','cKDTree','sklearnKDTree','pyKDTree', 'pyflannKDTree', 'cyflannKDTree'],
        'Generate KDTree (s)': [np.nanmin(pc_generate_KDTree_time)/inps.nr_of_repetitions, pc_generate_cKDTree_time.min()/inps.nr_of_repetitions, pc_generate_sklearnKDTree_time.min()/inps.nr_of_repetitions, pc_generate_pyKDTree_time.min()/inps.nr_of_repetitions, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions],
        'Query k=5 (s)': [pc_query_KDTree_k5_time.min()/inps.nr_of_repetitions, pc_query_cKDTree_k5_time.min()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time.min()/inps.nr_of_repetitions,  pc_query_pyKDTree_k5_time.min()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k5_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k5_time/inps.nr_of_repetitions],
        'Query k=10 (s)': [pc_query_KDTree_k10_time.min()/inps.nr_of_repetitions, pc_query_cKDTree_k10_time.min()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time.min()/inps.nr_of_repetitions,  pc_query_pyKDTree_k10_time.min()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k10_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k10_time/inps.nr_of_repetitions],
        'Query k=50 (s)': [pc_query_KDTree_k50_time.min()/inps.nr_of_repetitions, pc_query_cKDTree_k50_time.min()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time.min()/inps.nr_of_repetitions,  pc_query_pyKDTree_k50_time.min()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k50_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k50_time/inps.nr_of_repetitions],
        'Query k=100 (s)': [pc_query_KDTree_k100_time.min()/inps.nr_of_repetitions, pc_query_cKDTree_k100_time.min()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k100_time.min()/inps.nr_of_repetitions,  pc_query_pyKDTree_k100_time.min()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k100_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k100_time/inps.nr_of_repetitions],
        'Query k=500 (s)': [pc_query_KDTree_k500_time.min()/inps.nr_of_repetitions, pc_query_cKDTree_k500_time.min()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k500_time.min()/inps.nr_of_repetitions,  pc_query_pyKDTree_k500_time.min()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k500_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k500_time/inps.nr_of_repetitions],
        'Query k=1000 (s)': [pc_query_KDTree_k1000_time.min()/inps.nr_of_repetitions, pc_query_cKDTree_k1000_time.min()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k1000_time.min()/inps.nr_of_repetitions,  pc_query_pyKDTree_k1000_time.min()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k1000_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k1000_time/inps.nr_of_repetitions]
        }
    df_TimeTable_besttimes = pd.DataFrame(TimeTable_besttimes, columns= ['Algorithm', 'Generate KDTree (s)', 'Query k=5 (s)', 'Query k=10 (s)', 'Query k=50 (s)', 'Query k=100 (s)', 'Query k=500 (s)', 'Query k=1000 (s)'])
    print(df_TimeTable_besttimes)
    df_to_markdown(df_TimeTable_besttimes.round(2))

    TimeTable_worsttimes = {'Algorithm': ['KDTree','cKDTree','sklearnKDTree','pyKDTree', 'pyflannKDTree', 'cyflannKDTree'],
        'Generate KDTree (s)': [np.nanmax(pc_generate_KDTree_time)/inps.nr_of_repetitions, pc_generate_cKDTree_time.max()/inps.nr_of_repetitions, pc_generate_sklearnKDTree_time.max()/inps.nr_of_repetitions, pc_generate_pyKDTree_time.max()/inps.nr_of_repetitions, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions],
        'Query k=5 (s)': [pc_query_KDTree_k5_time.max()/inps.nr_of_repetitions, pc_query_cKDTree_k5_time.max()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time.max()/inps.nr_of_repetitions,  pc_query_pyKDTree_k5_time.max()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k5_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k5_time/inps.nr_of_repetitions],
        'Query k=10 (s)': [pc_query_KDTree_k10_time.max()/inps.nr_of_repetitions, pc_query_cKDTree_k10_time.max()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time.max()/inps.nr_of_repetitions,  pc_query_pyKDTree_k10_time.max()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k10_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k10_time/inps.nr_of_repetitions],
        'Query k=50 (s)': [pc_query_KDTree_k50_time.max()/inps.nr_of_repetitions, pc_query_cKDTree_k50_time.max()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time.max()/inps.nr_of_repetitions,  pc_query_pyKDTree_k50_time.max()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k50_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k50_time/inps.nr_of_repetitions],
        'Query k=100 (s)': [pc_query_KDTree_k100_time.max()/inps.nr_of_repetitions, pc_query_cKDTree_k100_time.max()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k100_time.max()/inps.nr_of_repetitions,  pc_query_pyKDTree_k100_time.max()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k100_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k100_time/inps.nr_of_repetitions],
        'Query k=500 (s)': [pc_query_KDTree_k500_time.max()/inps.nr_of_repetitions, pc_query_cKDTree_k500_time.max()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k500_time.max()/inps.nr_of_repetitions,  pc_query_pyKDTree_k500_time.max()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k500_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k500_time/inps.nr_of_repetitions],
        'Query k=1000 (s)': [pc_query_KDTree_k1000_time.max()/inps.nr_of_repetitions, pc_query_cKDTree_k1000_time.max()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k1000_time.max()/inps.nr_of_repetitions,  pc_query_pyKDTree_k1000_time.max()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k1000_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k1000_time/inps.nr_of_repetitions]
        }
    df_TimeTable_worsttimes = pd.DataFrame(TimeTable_worsttimes, columns= ['Algorithm', 'Generate KDTree (s)', 'Query k=5 (s)', 'Query k=10 (s)', 'Query k=50 (s)', 'Query k=100 (s)', 'Query k=500 (s)', 'Query k=1000 (s)'])
    print(df_TimeTable_worsttimes)
    df_to_markdown(df_TimeTable_worsttimes.round(2))

    TimeTable_rangetimes = {'Algorithm': ['KDTree','cKDTree','sklearnKDTree','pyKDTree', 'pyflannKDTree', 'cyflannKDTree'],
        'Generate KDTree (s)': [pc_generate_KDTree_time.ptp()/inps.nr_of_repetitions, pc_generate_cKDTree_time.ptp()/inps.nr_of_repetitions, pc_generate_sklearnKDTree_time.ptp()/inps.nr_of_repetitions, pc_generate_pyKDTree_time.ptp()/inps.nr_of_repetitions, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions],
        'Query k=5 (s)': [pc_query_KDTree_k5_time.ptp()/inps.nr_of_repetitions, pc_query_cKDTree_k5_time.ptp()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time.ptp()/inps.nr_of_repetitions,  pc_query_pyKDTree_k5_time.ptp()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k5_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k5_time/inps.nr_of_repetitions],
        'Query k=10 (s)': [pc_query_KDTree_k10_time.ptp()/inps.nr_of_repetitions, pc_query_cKDTree_k10_time.ptp()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time.ptp()/inps.nr_of_repetitions,  pc_query_pyKDTree_k10_time.ptp()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k10_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k10_time/inps.nr_of_repetitions],
        'Query k=50 (s)': [pc_query_KDTree_k50_time.ptp()/inps.nr_of_repetitions, pc_query_cKDTree_k50_time.ptp()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time.ptp()/inps.nr_of_repetitions,  pc_query_pyKDTree_k50_time.ptp()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k50_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k50_time/inps.nr_of_repetitions],
        'Query k=100 (s)': [pc_query_KDTree_k100_time.ptp()/inps.nr_of_repetitions, pc_query_cKDTree_k100_time.ptp()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k100_time.ptp()/inps.nr_of_repetitions,  pc_query_pyKDTree_k100_time.ptp()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k100_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k100_time/inps.nr_of_repetitions],
        'Query k=500 (s)': [pc_query_KDTree_k500_time.ptp()/inps.nr_of_repetitions, pc_query_cKDTree_k500_time.ptp()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k500_time.ptp()/inps.nr_of_repetitions,  pc_query_pyKDTree_k500_time.ptp()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k500_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k500_time/inps.nr_of_repetitions],
        'Query k=1000 (s)': [pc_query_KDTree_k1000_time.ptp()/inps.nr_of_repetitions, pc_query_cKDTree_k1000_time.ptp()/inps.nr_of_repetitions, pc_query_sklearnKDTree_k1000_time.ptp()/inps.nr_of_repetitions,  pc_query_pyKDTree_k1000_time.ptp()/inps.nr_of_repetitions, pc_query_pyflannKDTree_k1000_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k1000_time/inps.nr_of_repetitions]
        }
    df_TimeTable_rangetimes = pd.DataFrame(TimeTable_rangetimes, columns= ['Algorithm', 'Generate KDTree (s)', 'Query k=5 (s)', 'Query k=10 (s)', 'Query k=50 (s)', 'Query k=100 (s)', 'Query k=500 (s)', 'Query k=1000 (s)'])
    print(df_TimeTable_rangetimes)
    df_to_markdown(df_TimeTable_rangetimes.round(2))

    TimeTable_bestleafsize = {'Algorithm': ['KDTree','cKDTree','sklearnKDTree','pyKDTree'],
        'Generate KDTree (# leafsize)': [leafrange[np.argmin(pc_generate_KDTree_time)], leafrange[np.argmin(pc_generate_cKDTree_time)], leafrange[np.argmin(pc_generate_sklearnKDTree_time)], leafrange[np.argmin(pc_generate_pyKDTree_time)]],
        'Query k=5 (# leafsize)': [leafrange[np.argmin(pc_query_KDTree_k5_time)], leafrange[np.argmin(pc_query_cKDTree_k5_time)], leafrange[np.argmin(pc_query_sklearnKDTree_k5_time)],  leafrange[np.argmin(pc_query_pyKDTree_k5_time)]],
        'Query k=10 (# leafsize)': [leafrange[np.argmin(pc_query_KDTree_k10_time)], leafrange[np.argmin(pc_query_cKDTree_k10_time)], leafrange[np.argmin(pc_query_sklearnKDTree_k10_time)],  leafrange[np.argmin(pc_query_pyKDTree_k10_time)]],
        'Query k=50 (# leafsize)': [leafrange[np.argmin(pc_query_KDTree_k50_time)], leafrange[np.argmin(pc_query_cKDTree_k50_time)], leafrange[np.argmin(pc_query_sklearnKDTree_k50_time)],  leafrange[np.argmin(pc_query_pyKDTree_k50_time)]],
        'Query k=100 (# leafsize)': [leafrange[np.argmin(pc_query_KDTree_k100_time)], leafrange[np.argmin(pc_query_cKDTree_k100_time)], leafrange[np.argmin(pc_query_sklearnKDTree_k100_time)],  leafrange[np.argmin(pc_query_pyKDTree_k100_time)]],
        'Query k=500 (# leafsize)': [leafrange[np.argmin(pc_query_KDTree_k500_time)], leafrange[np.argmin(pc_query_cKDTree_k500_time)], leafrange[np.argmin(pc_query_sklearnKDTree_k500_time)],  leafrange[np.argmin(pc_query_pyKDTree_k500_time)]],
        'Query k=1000 (# leafsize)': [leafrange[np.argmin(pc_query_KDTree_k1000_time)], leafrange[np.argmin(pc_query_cKDTree_k1000_time)], leafrange[np.argmin(pc_query_sklearnKDTree_k1000_time)],  leafrange[np.argmin(pc_query_pyKDTree_k1000_time)]]
        }
    df_TimeTable_bestleafsize = pd.DataFrame(TimeTable_bestleafsize, columns= ['Algorithm', 'Generate KDTree (# leafsize)', 'Query k=5 (# leafsize)', 'Query k=10 (# leafsize)', 'Query k=50 (# leafsize)', 'Query k=100 (# leafsize)', 'Query k=500 (# leafsize)', 'Query k=1000 (# leafsize)'])
    print(df_TimeTable_bestleafsize)
    df_to_markdown(df_TimeTable_bestleafsize.round(2))
