# Modifies a given graph into every posible variant that may be relevant for the experiments

import sys
import os
from igraph import Graph
from graph_modification.graph_modification import *

if __name__ == "__main__":
    # This is to keep consistent results when modifing the graph
    seed_to_use = 4269

    dataset_name = sys.argv[1]
    dataset_location = sys.argv[2]
        
    graph = Graph.Read_Ncol(dataset_location+dataset_name+'/'+dataset_name+'_edges.txt', directed=False)

    # Create output directories if they don't exist
    os.makedirs(dataset_location+dataset_name+'/random', exist_ok=True)
    os.makedirs(dataset_location+dataset_name+'/bet_asc', exist_ok=True)
    os.makedirs(dataset_location+dataset_name+'/bet_desc', exist_ok=True)

    # The range of percentages to work with (0 is not taken into account)
    pcts = [x*0.1 for x in range(1,10)]

    # Create random removal of edges
    for pct in pcts:
        mod_graph = remove_edges(graph, pct=pct, deep_copy=True, strategy='random', order='asc', seed=seed_to_use)
        edge_list = []
        for edge in mod_graph.es():
            e_f, e_t = edge.tuple
            edge_list.append(graph.vs()[e_f]['name']+' '+graph.vs()[e_t]['name'])

        with open(dataset_location+dataset_name+'/random/'+str(int(pct*10)).zfill(2)+'.txt', 'w') as f:
            for line in edge_list:
                f.write("%s\n" % line)
        print('Finished creating ',dataset_name,'random',pct)

    # Create removal of edges with betweenness ascending
    for pct in pcts:
        mod_graph = remove_edges(graph, pct=pct, deep_copy=True, strategy='betweenness', order='asc', seed=seed_to_use)
        edge_list = []
        for edge in mod_graph.es():
            e_f, e_t = edge.tuple
            edge_list.append(graph.vs()[e_f]['name']+' '+graph.vs()[e_t]['name'])

        with open(dataset_location+dataset_name+'/bet_asc/'+str(int(pct*10)).zfill(2)+'.txt', 'w') as f:
            for line in edge_list:
                f.write("%s\n" % line)
        print('Finished creating ',dataset_name,'betweenness asc',pct)

    # Create removal of edges with betweenness ascending
    for pct in pcts:
        mod_graph = remove_edges(graph, pct=pct, deep_copy=True, strategy='betweenness', order='desc', seed=seed_to_use)
        edge_list = []
        for edge in mod_graph.es():
            e_f, e_t = edge.tuple
            edge_list.append(graph.vs()[e_f]['name']+' '+graph.vs()[e_t]['name'])

        with open(dataset_location+dataset_name+'/bet_desc/'+str(int(pct*10)).zfill(2)+'.txt', 'w') as f:
            for line in edge_list:
                f.write("%s\n" % line)
        print('Finished creating ',dataset_name,'betweenness desc',pct)