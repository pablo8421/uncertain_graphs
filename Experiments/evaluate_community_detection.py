import sys
import os
import igraph as ig
from igraph import Graph

import pickle
import random
import pandas as pd
import numpy as np

from sklearn.metrics import normalized_mutual_info_score
import community as community_louvain
import networkx as nx


if __name__ == "__main__":
	# This is to keep consistent results when modifing the graph
	seed_to_use = 4269

	dataset_name = sys.argv[1]
	graphs_folder = sys.argv[2]
	output_folder = sys.argv[3]
	community_file = sys.argv[4]

	# Load community belonging
	communities = {}
	with open(community_file) as f:
	    for line in f:
	       (key, val) = line.split()
	       communities[key] = val

	for subfolder in ['add_edges','add_weighted','add_all_weighted']:

		results_rows = []

		# Value to be used during all of the runs
		p_size = len(set(communities.values()))

		# Iterate generated graphs
		for file_name in os.listdir(graphs_folder+subfolder+'/'):
		    # Load graph
		    graph = Graph.Read_Ncol(graphs_folder+subfolder+'/'+file_name, directed=False)
		    
		    # Add community belonging
		    for vertex in graph.vs:
		        vertex['community'] = communities[vertex['name']]

		    # louvian
		    method = 'Louvain-igraph'

		    louvian = ig.Graph.community_multilevel(graph, return_levels=True)
		    louvian = louvian[len(louvian)-1]
		    p_louvian = len(set(louvian.membership))

		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P',
		                 'value':p_louvian})    
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P*/P',
		                 'value':p_size/p_louvian})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'Modularity',
		                 'value':graph.modularity(louvian.membership)})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'NMI',
		                 'value':normalized_mutual_info_score(graph.vs['community'], louvian.membership)})

		    # FastGreedy
		    method = 'Fastgreedy'

		    fg = ig.Graph.community_fastgreedy(graph)
		    p_fg = fg.optimal_count
		    fg = fg.as_clustering()
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P',
		                 'value':p_fg})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P*/P',
		                 'value':p_size/p_fg})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'Modularity',
		                 'value':graph.modularity(fg.membership)})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'NMI',
		                 'value':normalized_mutual_info_score(graph.vs['community'], fg.membership)})
		    # Infomap
		    method = 'Infomap'

		    infomap = ig.Graph.community_infomap(graph)
		    p_im = len(set(infomap.membership))
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P',
		                 'value':p_im})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P*/P',
		                 'value':p_size/p_im})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'Modularity',
		                 'value':graph.modularity(infomap.membership)})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'NMI',
		                 'value':normalized_mutual_info_score(graph.vs['community'], infomap.membership)})

		    # Label Propagation
		    method = 'Label Propagation'

		    lp = ig.Graph.community_label_propagation(graph)
		    p_lp = len(set(lp.membership))
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P',
		                 'value':p_lp})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P*/P',
		                 'value':p_size/p_lp})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'Modularity',
		                 'value':graph.modularity(lp.membership)})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'NMI',
		                 'value':normalized_mutual_info_score(graph.vs['community'], lp.membership)})

		    # Louvain
		    method = 'Louvain'

		    # Creating nx Graph to for other Louvain implementation
		    nxG = nx.Graph()
		    nxG.add_nodes_from([vertex.index for vertex in graph.vs])
		    nxG.add_edges_from([edge.tuple for edge in graph.es])

		    lv_partition = community_louvain.best_partition(nxG)
		    p_lv = len(set(lv_partition.values()))
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P',
		                 'value':p_lv})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'P*/P',
		                 'value':p_size/p_lv})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'Modularity',
		                 'value':graph.modularity(lv_partition.values())})
		    results_rows.append({'dataset':dataset_name, 'file_name':file_name, 'method':method, 'metric':'NMI',
		                 'value':normalized_mutual_info_score(graph.vs['community'], pd.Series(lv_partition.values()))})

		community_algos = pd.DataFrame(results_rows)

		community_algos['pct'] = community_algos['file_name'].apply(lambda x: 0 if x=='base.txt' else float(x[-5][:2])*0.1)
		community_algos['modification'] = community_algos['file_name'].apply(lambda x: 'random' if x.startswith('random') else 'bet_desc' if x.startswith('bet_desc') else 'bet_asc' if x.startswith('bet_asc') else 'base')
		community_algos.to_csv(output_folder+subfolder+'_communities.csv', index=False)