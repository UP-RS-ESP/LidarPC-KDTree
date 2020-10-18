#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bodo Bookhagen, bodo.bookhagen@uni-potsdam.de
Oct 2020
"""

import argparse, timeit, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cpuname = 'AMDRyzen_3900X_12cores'
pc_pyKDTree_time_df_12cores = pd.read_hdf(os.path.join('results_H5', 'pc_pyKDTree_time_df_%s.hdf'%(cpuname)), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_12cores = pd.read_hdf(os.path.join('results_H5','pc_cKDTree_time_df_%s.hdf'%(cpuname)), key='pc_cKDTree_time_df')
pc_cyflannKDTree_time_df_12cores = pd.read_hdf(os.path.join('results_H5','pc_cyflannKDTree_time_df_%s.hdf'%(cpuname)), key='pc_cyflannKDTree_time_df')
pc_pyflannKDTree_time_df_12cores = pd.read_hdf(os.path.join('results_H5','pc_pyflannKDTree_time_df_%s.hdf'%(cpuname)), key='pc_pyflannKDTree_time_df')

cpuname = 'AMDRyzen_2970WX_24cores'
pc_pyKDTree_time_df_24cores = pd.read_hdf(os.path.join('results_H5','pc_pyKDTree_time_df_%s.hdf'%(cpuname)), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_24cores = pd.read_hdf(os.path.join('results_H5','pc_cKDTree_time_df_%s.hdf'%(cpuname)), key='pc_cKDTree_time_df')
pc_cyflannKDTree_time_df_24cores = pd.read_hdf(os.path.join('results_H5','pc_cyflannKDTree_time_df_%s.hdf'%(cpuname)), key='pc_cyflannKDTree_time_df')
pc_pyflannKDTree_time_df_24cores = pd.read_hdf(os.path.join('results_H5','pc_pyflannKDTree_time_df_%s.hdf'%(cpuname)), key='pc_pyflannKDTree_time_df')

cpuname = 'Xeon_Gold6230_40cores'
pc_pyKDTree_time_df_40cores = pd.read_hdf(os.path.join('results_H5','pc_pyKDTree_time_df_%s.hdf'%(cpuname)), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_40cores = pd.read_hdf(os.path.join('results_H5','pc_cKDTree_time_df_%s.hdf'%(cpuname)), key='pc_cKDTree_time_df')
pc_cyflannKDTree_time_df_40cores = pd.read_hdf(os.path.join('results_H5','pc_cyflannKDTree_time_df_%s.hdf'%(cpuname)), key='pc_cyflannKDTree_time_df')
pc_pyflannKDTree_time_df_40cores = pd.read_hdf(os.path.join('results_H5','pc_pyflannKDTree_time_df_%s.hdf'%(cpuname)), key='pc_pyflannKDTree_time_df')

nr_of_repetitions = 3
nr_of_repetitions_generate = 1
leafrange = range(8,40,2)

fig = plt.figure()
ax = plt.subplot(111)
ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_12cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'x-', c='k', label='create + query cKDTree, 12 cores')
ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_24cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='darkblue', label='create + query cKDTree, 24 cores')
ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_40cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'s-', c='darkred', label='create + query cKDTree, 40 cores')
ax.set_title('cKDTree and 12, 24, 40 cores comparison with leaf size = 20')
ax.grid()
ax.set_xlabel('k (# of nearest neighbors)')
ax.set_ylabel('Query time (s)')
ax.set_xlim([0,1000])
#ax.set_yscale('log')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])
ax.legend()
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
#           fancybox=False, shadow=False, ncol=2,fontsize=8)
fig.savefig('docs/figs/pc_cKDTree_k5_to_k1000_vcores_leafsize20.png', dpi=300, orientation='landscape')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot([5,10,50,100,500,1000], [pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate],'x-', c='k', label='create + query pyKDTree, 12 cores')

ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_12cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='k', label='create + query cKDTree, 12 cores')

ax.plot([5,10,50,100,500,1000], [pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate],'x-', c='darkblue', label='create + query pyKDTree, 24 cores')

ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_24cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='darkblue', label='create + query cKDTree, 24 cores')

ax.plot([5,10,50,100,500,1000], [pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate],'x-', c='darkred', label='create + query pyKDTree, 40 cores')

ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_40cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='darkred', label='create + query cKDTree, 40 cores')
ax.set_title('pyKDTree+cKDTree and 12, 24, 40 cores comparison with leaf size = 20')
ax.grid()
ax.set_xlabel('k (# of nearest neighbors)')
ax.set_ylabel('Query time (s)')
ax.set_xlim([0,1000])
#ax.set_yscale('log')
ax.legend()
fig.savefig('docs/figs/pc_ckDTree_pyKDTree_k5_to_k1000_vcores_leafsize20.png', dpi=300, orientation='landscape')


# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot([5,10,50,100,500,1000], [pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate],'x-', c='k', label='create + query pyKDTree, 12 cores')
#
# ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_12cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_12cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_12cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_12cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_12cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_12cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'x-', c='darkblue', label='create + query cKDTree, 12 cores')
#
# ax.plot([5,10,50,100,500,1000], [pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='k', label='create + query pyKDTree, 24 cores')
#
# ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_24cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_24cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_24cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_24cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_24cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_24cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='darkblue', label='create + query cKDTree, 24 cores')
#
# ax.plot([5,10,50,100,500,1000], [pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time'].iloc[6]/nr_of_repetitions_generate],'s-', c='k', label='create + query pyKDTree, 40 cores')
#
# ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_40cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_40cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_40cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_40cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_40cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
#                                 pc_cKDTree_time_df_40cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'s-', c='darkblue', label='create + query cKDTree, 40 cores')
# ax.set_title('pyKDTree+cKDTree and various cores comparison')
# ax.grid()
# ax.set_xlabel('k (# of nearest neighbors)')
# ax.set_ylabel('Query time (s)')
# ax.set_xlim([0,1000])
# #ax.set_yscale('log')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.3,
#                  box.width, box.height * 0.7])
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
#           fancybox=False, shadow=False, ncol=2,fontsize=8)
# fig.savefig('docs/figs/pc_ckDTree_pyKDTree_k5_to_k1000_vcores_leafsize20.png', dpi=300, orientation='landscape')


# ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k5_time'])/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 'o-', c='k', label='create + query pyKDTree k=50, 24 cores')
# ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k5_time'])/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 's-', c='k', label='create + query pyKDTree k=50, 40 cores')
# ax.plot(leafrange, (pc_cKDTree_time_df_12cores['pc_query_cKDTree_k5_time'])/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time']/nr_of_repetitions_generate, 'x-', c='darkblue', label='create + query cKDTree k=50, 12 cores')
# ax.plot(leafrange, (pc_cKDTree_time_df_24cores['pc_query_cKDTree_k5_time'])/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time']/nr_of_repetitions_generate, 'o-', c='darkblue', label='create + query cKDTree k=50, 24 cores')
# ax.plot(leafrange, (pc_cKDTree_time_df_40cores['pc_query_cKDTree_k5_time'])/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time']/nr_of_repetitions_generate, 's-', c='darkblue', label='create + query cKDTree k=50, 40 cores')
#
# ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k50_time'])/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 'x-', c='darkblue', label='create + query pyKDTree k=50, 12 cores')
# ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k50_time'])/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 'o-', c='darkblue', label='create + query pyKDTree k=50, 24 cores')
# ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k50_time'])/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 's-', c='darkblue', label='create + query pyKDTree k=50, 40 cores')
# ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k500_time'])/nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 'x-', c='darkred', label='create + query pyKDTree k=500, 12 cores')
# ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k500_time'])/nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 'o-', c='darkred', label='create + query pyKDTree k=500, 24 cores')
# ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k500_time'])/nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/nr_of_repetitions_generate, 's-', c='darkred', label='create + query pyKDTree k=500, 40 cores')
# ax.plot(leafrange, (pc_cKDTree_time_df_12cores['pc_query_cKDTree_k500_time'])/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time']/nr_of_repetitions_generate, 'x-', c='red', label='create + query cKDTree k=500, 12 cores')
# ax.plot(leafrange, (pc_cKDTree_time_df_24cores['pc_query_cKDTree_k500_time'])/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time']/nr_of_repetitions_generate, 'o-', c='red', label='create + query cKDTree k=500, 24 cores')
# ax.plot(leafrange, (pc_cKDTree_time_df_40cores['pc_query_cKDTree_k500_time'])/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time']/nr_of_repetitions_generate, 's-', c='red', label='create + query cKDTree k=500, 40 cores')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot([5,10,50,100,500,1000], [pc_cyflannKDTree_time_df_12cores['pc_query_cyflannKDTree_k5_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_12cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_12cores['pc_query_cyflannKDTree_k10_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_12cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_12cores['pc_query_cyflannKDTree_k50_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_12cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_12cores['pc_query_cyflannKDTree_k100_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_12cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_12cores['pc_query_cyflannKDTree_k500_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_12cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_12cores['pc_query_cyflannKDTree_k1000_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_12cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate],'x-', c='k', label='create + query cyFLANN, 12 cores')

ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_12cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_12cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='k', label='create + query cKDTree, 12 cores')

ax.plot([5,10,50,100,500,1000], [pc_cyflannKDTree_time_df_24cores['pc_query_cyflannKDTree_k5_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_24cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_24cores['pc_query_cyflannKDTree_k10_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_24cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_24cores['pc_query_cyflannKDTree_k50_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_24cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_24cores['pc_query_cyflannKDTree_k100_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_24cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_24cores['pc_query_cyflannKDTree_k500_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_24cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_24cores['pc_query_cyflannKDTree_k1000_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_24cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate],'x-', c='darkblue', label='create + query cyFLANN, 24 cores')

ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_24cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_24cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='darkblue', label='create + query cKDTree, 24 cores')

ax.plot([5,10,50,100,500,1000], [pc_cyflannKDTree_time_df_40cores['pc_query_cyflannKDTree_k5_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_40cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_40cores['pc_query_cyflannKDTree_k10_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_40cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_40cores['pc_query_cyflannKDTree_k50_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_40cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_40cores['pc_query_cyflannKDTree_k100_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_40cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_40cores['pc_query_cyflannKDTree_k500_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_40cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate,
                                pc_cyflannKDTree_time_df_40cores['pc_query_cyflannKDTree_k1000_time']/nr_of_repetitions+pc_cyflannKDTree_time_df_40cores['pc_generate_cyflannKDTree_time']/nr_of_repetitions_generate],'x-', c='darkred', label='create + query cyFLANN, 40 cores')

ax.plot([5,10,50,100,500,1000], [pc_cKDTree_time_df_40cores['pc_query_cKDTree_k5_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k10_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k50_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k100_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k500_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate,
                                pc_cKDTree_time_df_40cores['pc_query_cKDTree_k1000_time'].iloc[6]/nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time'].iloc[6]/nr_of_repetitions_generate],'o-', c='darkred', label='create + query cKDTree, 40 cores')
ax.set_title('cKDTree and cyFLANN for 12, 24, 40 cores comparison with leaf size = 20')
ax.grid()
ax.set_xlabel('k (# of nearest neighbors)')
ax.set_ylabel('Query time (s)')
ax.set_xlim([0,1000])
#ax.set_yscale('log')
ax.legend()
fig.savefig('docs/figs/pc_ckDTree_cyfFLANN_k5_to_k1000_vcores_leafsize20.png', dpi=300, orientation='landscape')
