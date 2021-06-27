import igraph as ig
from igraph import Graph
import numpy as np
import math
import random
import struct
import copy

def remove_edges(graph: Graph, strategy:str = 'random', order:str = 'desc', pct:float = None, k:float = 1, deep_copy:bool = False, seed:int=None):
    edges_strategies = ['random', 'betweenness']

    # Error checking before actual execution
    if strategy not in edges_strategies:
        raise ValueError('Strategy '+strategy+' not in the list of proper strategies')

    # Setting seed if given
    if seed != None:
        random.seed(seed)

    # Create deep copy if desired
    if deep_copy:
        graph = copy.deepcopy(graph)

    # Calculating the percentage of edges to remove
    if pct != None:
        if not(0.0 < pct < 1.0):
            raise ValueError(str(pct)+' not in 0 to 1 range')

        k = int(graph.ecount() * pct)
    
    if strategy == 'random':
        weights = None
    elif strategy == 'betweenness':
        weights = graph.edge_betweenness()
        weights = [float(x) / sum(weights) for x in weights]
        if order == 'asc':
            weights = [1 - x for x in weights]
        weights = [x / sum(weights) for x in weights]

    # List of tuples of edges to delete
    edges_to_delete = [edge.tuple for edge in np.random.choice(graph.es, size = k, replace=False,p=weights)]

    # Actual deletion of edges
    graph.delete_edges(edges_to_delete)

    return graph

def remove_vertices(graph: Graph, strategy:str = 'random', order:str = 'desc', pct:float = None, k:float = 1, deep_copy:bool = False, seed:int=None):
    vertices_strategies = ['random', 'degree', 'betweenness', 'closeness','clustering_coefficient']

    # Error checking before actual execution
    if strategy not in vertices_strategies:
        raise ValueError('Strategy '+strategy+' not in the list of proper strategies')

    # Setting seed if given
    if seed != None:
        random.seed(seed)

    # Create deep copy if desired
    if deep_copy:
        graph = copy.deepcopy(graph)

    # Calculating the percentage of edges to remove
    if pct != None:
        if not(0.0 < pct < 1.0):
            raise ValueError(str(pct)+' not in 0 to 1 range')

        k = int(graph.vcount() * pct)
    
    if strategy == 'random':
        weights = None
    else:
        if strategy == 'degree':
            weights = graph.degree()
        elif strategy == 'betweenness':
            weights = graph.betweenness()
        elif strategy == 'closeness':
            weights = graph.closeness()
        elif strategy == 'clustering_coefficient':
            weights = graph.transitivity_local_undirected()
            weights = [0.0 if math.isnan(x) else x for x in weights] 

        #Set values to sum 1
        weights = [float(x) / sum(weights) for x in weights]

        # 'Inverse' probabilities if it's asked for
        if order == 'asc':
            weights = [1.0 - x for x in weights]
            weights = [x / sum(weights) for x in weights]

    # List of tuples of edges to delete
    vertices_to_delete = np.random.choice(graph.vs, size = k, replace=False,p=weights)

    # Actual deletion of edges
    graph.delete_vertices(vertices_to_delete)

    return graph

if __name__ == "__main__":
    datasets_location = 'D:/Pablo/clases/UJM/2. Semester, 2021/Mining Uncertain Social Networks/Graphs Visualization/datasets/'
    graph_file = 'citeseer/Citeseer_graph.txt'

    g = Graph.Load(datasets_location+graph_file, format='pajek')

    print(g.vcount()) # Vertex count
    print(g.ecount()) # Edge count
    print('------------------------------')
    n_g = remove_vertices(g, pct=0.5, deep_copy=True, strategy='degree', order='desc')
    #n_g = remove_edges(g, k=4536, deep_copy=True, strategy='random', order='asc', seed=546)
    print('------------------------------')
    print(n_g.vcount()) # Vertex count
    print(n_g.ecount()) # Edge count