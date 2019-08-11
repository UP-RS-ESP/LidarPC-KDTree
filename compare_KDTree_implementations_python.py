#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bodo Bookhagen, bodo.bookhagen@uni-potsdam.de
August 2019

Example code that will compare the common KD-Tree implementation in
Python.
The code times the KD-Tree generation and then the query of 
k=5,10,50,100,500,1000 nearest neighbors.

"""
#conda install tabulate pandas pytables scipy pykdtree pyflann cyflann

import argparse, numpy as np, pandas as pd, timeit

def pc_generate_KDTree(pc_xyz):
    try:
        from scipy import spatial
    except ImportError:
        raise pc_generate_KDTree("scipy not installed.")
    pc_xyz_KDTree_tree = spatial.KDTree(pc_xyz)
    return pc_xyz_KDTree_tree
    
def pc_query_KDTree(pc_xyz_KDTree_tree, pc_xyz, k=10):
    pc_kdtree_distance, pc_kdtree_id = pc_xyz_KDTree_tree.query(pc_xyz, k=k)
    return pc_kdtree_distance, pc_kdtree_id

def pc_generate_cKDTree(pc_xyz):
    try:
        from scipy import spatial
    except ImportError:
        raise pc_generate_cKDTree("scipy not installed.")
    pc_xyz_cKDTree_tree = spatial.cKDTree(pc_xyz)
    return pc_xyz_cKDTree_tree
    
def pc_query_cKDTree(pc_xyz_cKDTree_tree, pc_xyz, k=10):
    pc_cKDTree_distance, pc_cKDTree_id = pc_xyz_cKDTree_tree.query(pc_xyz, k=k)
    return pc_cKDTree_distance, pc_cKDTree_id

def pc_generate_pyKDTree(pc_xyz):
    try:
        from pykdtree.kdtree import KDTree as pyKDTree
    except ImportError:
        raise pc_generate_pyKDTree("pykdtree not installed.")
    pc_xyz_pyKDTree_tree = pyKDTree(pc_xyz)
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
    pc_xyz_pyflannKDTree_tree.build_index(pc_xyz, algorithm='kdtree_single')
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
    pc_xyz_cyflannKDTree_tree.build_index(pc_xyz, algorithm='kdtree_single')
    return pc_xyz_cyflannKDTree_tree
    
def pc_query_cyflannKDTree(pc_xyz_cyflannKDTree_tree, pc_xyz, k=10):
    pc_cyflannKDTree_id, pc_cyflannKDTree_distance = pc_xyz_cyflannKDTree_tree.nn_index(pc_xyz, k)
    return pc_cyflannKDTree_distance, pc_cyflannKDTree_id

def pc_generate_sklearnKDTree(pc_xyz):
    #conda install scikit-learn
    try:
        from sklearn.neighbors import KDTree as sklearnKDTree
    except ImportError:
        raise pc_generate_sklearnKDTree("sklearn not installed.")
    pc_xyz_sklearnKDTree_tree = sklearnKDTree(pc_xyz)
    return pc_xyz_sklearnKDTree_tree
    
def pc_query_sklearnKDTree(pc_xyz_sklearnKDTree_tree, pc_xyz, k=10):
    pc_sklearnKDTree_id, pc_sklearnKDTree_distance = pc_xyz_sklearnKDTree_tree.query(pc_xyz, k=k)
    return pc_sklearnKDTree_distance, pc_sklearnKDTree_id

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
    inps = argparse.ArgumentParser(description='Compare KDTree algorithms for lidar or SfM PointClouds (PC). B. Bookhagen (bodo.bookhagen@uni-potsdam.de), Aug 2019.')
    inps.inlas = 'Pozo_WestCanada_clg.laz'
    inps.nr_of_repetitions = 5
    inps.hdf_filename = 'ThinkPad_P50.hdf'
    inps.csv_filename = 'ThinkPad_P50_repetitions5.csv'
    
    print('Loading input file: %s... '%inps.inlas, end='', flush=True)
    pc_xyz = load_LAS(inps.inlas, dtype='float32')
    print('loaded %s points'%"{:,}".format(pc_xyz.shape[0]))

    #Run standard KDTree (python implementation from scipy)
    ## Not RUNNING KDTREE querying, because it is too slow.
    print('\n\tGenerating KDTree... (%dx)'%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_generate_KDTree, pc_xyz)
    pc_generate_KDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    pc_xyz_KDTree_tree = pc_generate_KDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_generate_KDTree_time/inps.nr_of_repetitions, pc_generate_KDTree_time/inps.nr_of_repetitions/60))

#    print('\tQuerying KDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=5)
#    pc_query_KDTree_k5_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k5_time/inps.nr_of_repetitions, pc_query_KDTree_k5_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying KDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_KDTree, pc_xyz_KDTree_tree, pc_xyz, k=10)
#    pc_query_KDTree_k10_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_KDTree_k10_time/inps.nr_of_repetitions, pc_query_KDTree_k10_time/inps.nr_of_repetitions/60))
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

    #Run cKDTree (cython implementation from scipy)
    print('\n\tGenerating cKDTree... (%dx) '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_generate_cKDTree, pc_xyz)
    pc_generate_cKDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    pc_xyz_cKDTree_tree = pc_generate_cKDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_generate_cKDTree_time/inps.nr_of_repetitions, pc_generate_cKDTree_time/inps.nr_of_repetitions/60))

    print('\tQuerying cKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=5)
    pc_query_cKDTree_k5_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k5_time/inps.nr_of_repetitions, pc_query_cKDTree_k5_time/inps.nr_of_repetitions/60))

    print('\tQuerying cKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=10)
    pc_query_cKDTree_k10_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k10_time/inps.nr_of_repetitions, pc_query_cKDTree_k10_time/inps.nr_of_repetitions/60))

    print('\tQuerying cKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=50)
    pc_query_cKDTree_k50_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k50_time/inps.nr_of_repetitions, pc_query_cKDTree_k50_time/inps.nr_of_repetitions/60))

#    print('\tQuerying cKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=100)
#    pc_query_cKDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k100_time/inps.nr_of_repetitions, pc_query_cKDTree_k100_time/inps.nr_of_repetitions/60))
#    
#    print('\tQuerying cKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=500)
#    pc_query_cKDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k500_time/inps.nr_of_repetitions, pc_query_cKDTree_k500_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying cKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_cKDTree, pc_xyz_cKDTree_tree, pc_xyz, k=1000)
#    pc_query_cKDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cKDTree_k1000_time/inps.nr_of_repetitions, pc_query_cKDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_cKDTree_tree = None

    #Run sklearnKDTree 
    print('\n\tGenerating sklearnKDTree... (%dx) '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_generate_sklearnKDTree, pc_xyz)
    pc_generate_sklearnKDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    pc_xyz_sklearnKDTree_tree = pc_generate_sklearnKDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_generate_sklearnKDTree_time/inps.nr_of_repetitions, pc_generate_sklearnKDTree_time/inps.nr_of_repetitions/60))

    print('\tQuerying sklearnKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=5)
    pc_query_sklearnKDTree_k5_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time/inps.nr_of_repetitions/60))

    print('\tQuerying sklearnKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=10)
    pc_query_sklearnKDTree_k10_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time/inps.nr_of_repetitions/60))

    print('\tQuerying sklearnKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=50)
    pc_query_sklearnKDTree_k50_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time/inps.nr_of_repetitions/60))

#    print('\tQuerying sklearnKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=100)
#    pc_query_sklearnKDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k100_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k100_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying sklearnKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=500)
#    pc_query_sklearnKDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k500_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k500_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying sklearnKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_sklearnKDTree, pc_xyz_sklearnKDTree_tree, pc_xyz, k=1000)
#    pc_query_sklearnKDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_sklearnKDTree_k1000_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_sklearnKDTree_tree = None

    #Run pyKDTree (fast implementation using c and libANN)
    print('\n\tGenerating pyKDTree... (%dx) '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_generate_pyKDTree, pc_xyz)
    pc_generate_pyKDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    pc_xyz_pyKDTree_tree = pc_generate_pyKDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_generate_pyKDTree_time/inps.nr_of_repetitions, pc_generate_pyKDTree_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyKDTree with k=5 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=5)
    pc_query_pyKDTree_k5_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k5_time/inps.nr_of_repetitions, pc_query_pyKDTree_k5_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyKDTree with k=10 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=10)
    pc_query_pyKDTree_k10_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k10_time/inps.nr_of_repetitions, pc_query_pyKDTree_k10_time/inps.nr_of_repetitions/60))

    print('\tQuerying pyKDTree with k=50 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=50)
    pc_query_pyKDTree_k50_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k50_time/inps.nr_of_repetitions, pc_query_pyKDTree_k50_time/inps.nr_of_repetitions/60))

#    print('\tQuerying pyKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=100)
#    pc_query_pyKDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k100_time/inps.nr_of_repetitions, pc_query_pyKDTree_k100_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying pyKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=500)
#    pc_query_pyKDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k500_time/inps.nr_of_repetitions, pc_query_pyKDTree_k500_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying pyKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_pyKDTree, pc_xyz_pyKDTree_tree, pc_xyz, k=1000)
#    pc_query_pyKDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyKDTree_k1000_time/inps.nr_of_repetitions, pc_query_pyKDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_pyKDTree_tree = None

    #Run pyflannKDTree 
    print('\n\tGenerating pyflannKDTree... (%dx) '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_generate_pyflannKDTree, pc_xyz)
    pc_generate_pyflannKDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    pc_xyz_pyflannKDTree_tree = pc_generate_pyflannKDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions/60))

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

#    print('\tQuerying pyflannKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=100)
#    pc_query_pyflannKDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k100_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k100_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying pyflannKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=500)
#    pc_query_pyflannKDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k500_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k500_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying pyflannKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_pyflannKDTree, pc_xyz_pyflannKDTree_tree, pc_xyz, k=1000)
#    pc_query_pyflannKDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_pyflannKDTree_k1000_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_pyflannKDTree_tree = None

    #Run cyflannKDTree 
    print('\n\tGenerating cyflannKDTree... (%dx) '%(inps.nr_of_repetitions), end='', flush=True)
    wrapped = wrapper(pc_generate_cyflannKDTree, pc_xyz)
    pc_generate_cyflannKDTree_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
    pc_xyz_cyflannKDTree_tree = pc_generate_cyflannKDTree(pc_xyz)
    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions/60))

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

#    print('\tQuerying cyflannKDTree with k=100 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=100)
#    pc_query_cyflannKDTree_k100_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k100_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k100_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying cyflannKDTree with k=500 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=500)
#    pc_query_cyflannKDTree_k500_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k500_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k500_time/inps.nr_of_repetitions/60))
#
#    print('\tQuerying cyflannKDTree with k=1000 for all points (%dx)... '%(inps.nr_of_repetitions), end='', flush=True)
#    wrapped = wrapper(pc_query_cyflannKDTree, pc_xyz_cyflannKDTree_tree, pc_xyz, k=1000)
#    pc_query_cyflannKDTree_k1000_time = timeit.timeit(wrapped, number=inps.nr_of_repetitions)
#    print('time (average of %d runs): %0.3fs or %0.2fm'%(inps.nr_of_repetitions, pc_query_cyflannKDTree_k1000_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k1000_time/inps.nr_of_repetitions/60))
    pc_xyz_cyflannKDTree_tree = None

    ## CUDA Processing
    # Fit a NearestNeighbors model and query it
    import cudf
    from cuml.neighbors.nearest_neighbors import NearestNeighbors as cumlKNN
    from sklearn.neighbors import NearestNeighbors as skKNN
#    nrows=pc_xyz.shape[0]
#    ncols=pc_xyz.shape[1]
#    X = np.random.random((nrows,ncols)).astype('float32')
#    df = pd.DataFrame({'fea%d'%i:X[:,i] for i in range(X.shape[1])}).fillna(0)
    df = pd.DataFrame({'X':pc_xyz[:,0], 'Y':pc_xyz[:,1], 'Z':pc_xyz[:,2],})

    n_neighbors = 5
    # use the sklearn KNN model to fit the dataset 
    knn_sk = skKNN(metric = 'sqeuclidean', )
    knn_sk.fit(pc)
    D_sk,I_sk = knn_sk.kneighbors(X,n_neighbors)
    %%time
    # convert the pandas dataframe to cudf dataframe
    X = cudf.DataFrame.from_pandas(X)

    # use cuml's KNN model to fit the dataset
    knn_cuml = cumlKNN()
    knn_cuml.fit(X)
    
    # calculate the distance and the indices of the samples present in the dataset
    D_cuml,I_cuml = knn_cuml.kneighbors(X,n_neighbors)

    TimeTable = {'Algorithm': ['KDTree','cKDTree','sklearnKDTree','pyKDTree', 'pyflannKDTree', 'cyflannKDTree'],
        'Generate KDTree (s)': [pc_generate_KDTree_time/inps.nr_of_repetitions, pc_generate_cKDTree_time/inps.nr_of_repetitions, pc_generate_sklearnKDTree_time/inps.nr_of_repetitions, pc_generate_pyKDTree_time/inps.nr_of_repetitions, pc_generate_pyflannKDTree_time/inps.nr_of_repetitions, pc_generate_cyflannKDTree_time/inps.nr_of_repetitions],
        'Query k=5 (s)': [pc_query_KDTree_k5_time/inps.nr_of_repetitions, pc_query_cKDTree_k5_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k5_time/inps.nr_of_repetitions,  pc_query_pyKDTree_k5_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k5_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k5_time/inps.nr_of_repetitions],
        'Query k=10 (s)': [pc_query_KDTree_k10_time/inps.nr_of_repetitions, pc_query_cKDTree_k10_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k10_time/inps.nr_of_repetitions,  pc_query_pyKDTree_k10_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k10_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k10_time/inps.nr_of_repetitions],
        'Query k=50 (s)': [pc_query_KDTree_k50_time/inps.nr_of_repetitions, pc_query_cKDTree_k50_time/inps.nr_of_repetitions, pc_query_sklearnKDTree_k50_time/inps.nr_of_repetitions,  pc_query_pyKDTree_k50_time/inps.nr_of_repetitions, pc_query_pyflannKDTree_k50_time/inps.nr_of_repetitions, pc_query_cyflannKDTree_k50_time/inps.nr_of_repetitions]
        }

    df_TimeTable = pd.DataFrame(TimeTable, columns= ['Algorithm', 'Generate KDTree (s)', 'Query k=5 (s)', 'Query k=10 (s)', 'Query k=50 (s)'])
    print(df_TimeTable)
    df_TimeTable.to_csv(inps.csv_filename, ',')
    df_to_markdown(df_TimeTable.round(2))
    