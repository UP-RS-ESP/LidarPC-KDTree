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

inps.cpuname= 'AMDRyzen_3900X_12cores'
pc_pyKDTree_time_df_12cores = pd.read_hdf('pc_pyKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_12cores = pd.read_hdf('pc_cKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cKDTree_time_df')

inps.cpuname= 'AMDRyzen_2970WX_24cores'
pc_pyKDTree_time_df_24cores = pd.read_hdf('pc_pyKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_24cores = pd.read_hdf('pc_cKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cKDTree_time_df')

inps.cpuname= 'Xeon_Gold6230_40cores'
pc_pyKDTree_time_df_40cores = pd.read_hdf('pc_pyKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_40cores = pd.read_hdf('pc_cKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cKDTree_time_df')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k5_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='k', label='create + query pyKDTree k=5, 12 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k5_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='k', label='create + query pyKDTree k=5, 24 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k5_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='k', label='create + query pyKDTree k=5, 40 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k50_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='darkblue', label='create + query pyKDTree k=50, 12 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k50_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='darkblue', label='create + query pyKDTree k=50, 24 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k50_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='darkblue', label='create + query pyKDTree k=50, 40 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k500_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='darkred', label='create + query pyKDTree k=500, 12 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k500_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='darkred', label='create + query pyKDTree k=500, 24 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k500_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='darkred', label='create + query pyKDTree k=500, 40 cores')
ax.plot(leafrange, (pc_cKDTree_time_df_12cores['pc_query_cKDTree_k500_time'])/inps.nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='red', label='create + query cKDTree k=500, 12 cores')
ax.plot(leafrange, (pc_cKDTree_time_df_24cores['pc_query_cKDTree_k500_time'])/inps.nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='red', label='create + query cKDTree k=500, 24 cores')
ax.plot(leafrange, (pc_cKDTree_time_df_40cores['pc_query_cKDTree_k500_time'])/inps.nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='red', label='create + query cKDTree k=500, 40 cores')
ax.set_title('pyKDTree+cKDTree and various cores comparison')
ax.grid()
ax.set_xlabel('leafsize')
ax.set_ylabel('Query time (s)')
ax.set_xlim([8,30])
ax.set_xticks(np.arange(8,30,step=2))
ax.set_yscale('log')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.7])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=False, shadow=False, ncol=2,fontsize=8)
fig.savefig('figs/pc_pyKDTree_k5_k50_vcores.png', dpi=300, orientation='landscape')
