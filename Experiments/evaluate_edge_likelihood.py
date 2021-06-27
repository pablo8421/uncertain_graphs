# Loads and evaluates the edge likelihood function for all of the variations of the dataset

import sys
import pickle
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import itertools
from asymproj_edge_dnn import edge_nn

if __name__ == "__main__":
    # This is to keep consistent results when modifing the graph
    seed_to_use = 4269

    dataset_name = sys.argv[1]
    datasets_location = sys.argv[2]
        
    # Load the original edges of the graph
    original_edges = pd.read_csv(datasets_location+dataset_name+'/'+dataset_name+'_edges.txt', header=None, sep=' ', dtype=int)
    original_edges['from'] = original_edges[[0,1]].min(axis=1)
    original_edges['to'] = original_edges[[0,1]].max(axis=1)
    original_edges['edge_exists_original'] = 1
    original_edges = original_edges[['from','to','edge_exists_original']]

    # Base case (no modification)
    # Preconfiguration of the tensorflow session
    tf.keras.backend.clear_session()
    tf.compat.v1.disable_eager_execution()

    #Loading embedding, network parameters and the mapping of the nodes
    embeddings = np.load(datasets_location+dataset_name+'/base/dumps/test.d100_f100x100_g32_embeddings.npy.best')
    net_values = dict(pickle.load(tf.gfile.Open(datasets_location+dataset_name+'/base/dumps/test.d100_f100x100_g32_net.pkl.best', 'rb')))
    node_index = np.load(datasets_location+dataset_name+'/base/index.pkl', allow_pickle=True)['index']
    inv_node_index = {v: k for k, v in node_index.items()}

    # Create the edge Neural Network.
    nn = edge_nn.EdgeNN()
    nn.build_net(
      embedding_dim=100, dnn_dims='100,100',
      projection_dim=32,
      num_projections=1)

    # Create tensorflow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Load nn values
    for v in tf.global_variables():
        if v.name in net_values:
            sess.run(v.assign(net_values[v.name]))

    # Create all posible edges combination
    all_edges = np.array(list(itertools.combinations(
                            [x for x in range(0,len(embeddings))], 2)))

    # Get edge likelihood for all of the posible edges
    edge_likelihood = sess.run(
        nn.output,
        feed_dict={
          nn.embeddings_a: embeddings[all_edges[:, 0]],
          nn.embeddings_b: embeddings[all_edges[:, 1]],
          nn.batch_size: len(all_edges),
        })

    # Create datafrom from likelihood
    all_edges_likelihood = pd.DataFrame()

    all_edges_likelihood['first'] = [int(inv_node_index[x]) for x in all_edges[:, 0]]
    all_edges_likelihood['second'] = [int(inv_node_index[x]) for x in all_edges[:, 1]]

    all_edges_likelihood['from'] = all_edges_likelihood[['first','second']].min(axis=1)
    all_edges_likelihood['to'] = all_edges_likelihood[['first','second']].max(axis=1)


    #Likelihood transformed into standard logistic, acording to equation 8 in the paper
    all_edges_likelihood['likelihood'] = 1/(1+np.exp(-edge_likelihood))
    all_edges_likelihood = all_edges_likelihood[['from','to','likelihood']]

    # Load the modified edges of the graph
    modified_edges = pd.read_csv(datasets_location+dataset_name+'/'+dataset_name+'_edges.txt', header=None, sep=' ', dtype=int)
    modified_edges['from'] = modified_edges[[0,1]].min(axis=1)
    modified_edges['to'] = modified_edges[[0,1]].max(axis=1)

    modified_edges['edge_exists_modified'] = 1
    modified_edges = modified_edges[['from','to','edge_exists_modified']]

    # Combine information and save results
    results = pd.merge(all_edges_likelihood, original_edges, on=['from','to'], how='left')
    results = pd.merge(results, modified_edges, on=['from','to'], how='left')

    results['edge_exists_original'] = results['edge_exists_original'].fillna(0)
    results['edge_exists_modified'] = results['edge_exists_modified'].fillna(0)
    results['dataset'] = dataset_name
    results['modification'] = 'base'

    results.to_csv(datasets_location+dataset_name+'/results/base.txt', index=False)


    # Modification cases
    removal_types = ['random', 'bet_asc', 'bet_desc']

    for rem_type in removal_types:
        for i in range (1,10):

            # Preconfiguration of the tensorflow session
            tf.keras.backend.clear_session()
            tf.compat.v1.disable_eager_execution()

            #Loading embedding, network parameters and the mapping of the nodes
            embeddings = np.load(datasets_location+dataset_name+'/'+rem_type+'/0'+str(i)+'/dumps/test.d100_f100x100_g32_embeddings.npy.best')
            net_values = dict(pickle.load(tf.gfile.Open(datasets_location+dataset_name+'/'+rem_type+'/0'+str(i)+'/dumps/test.d100_f100x100_g32_net.pkl.best', 'rb')))
            node_index = np.load(datasets_location+dataset_name+'/'+rem_type+'/0'+str(i)+'/index.pkl', allow_pickle=True)['index']
            inv_node_index = {v: k for k, v in node_index.items()}

            # Create the edge Neural Network.
            nn = edge_nn.EdgeNN()
            nn.build_net(
              embedding_dim=100, dnn_dims='100,100',
              projection_dim=32,
              num_projections=1)

            # Create tensorflow session
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            # Load nn values
            for v in tf.global_variables():
                if v.name in net_values:
                    sess.run(v.assign(net_values[v.name]))

            # Create all posible edges combination
            all_edges = np.array(list(itertools.combinations(
                                    [x for x in range(0,len(embeddings))], 2)))

            # Get edge likelihood for all of the posible edges
            edge_likelihood = sess.run(
                nn.output,
                feed_dict={
                  nn.embeddings_a: embeddings[all_edges[:, 0]],
                  nn.embeddings_b: embeddings[all_edges[:, 1]],
                  nn.batch_size: len(all_edges),
                })

            # Create datafrom from likelihood
            all_edges_likelihood = pd.DataFrame()

            all_edges_likelihood['first'] = [int(inv_node_index[x]) for x in all_edges[:, 0]]
            all_edges_likelihood['second'] = [int(inv_node_index[x]) for x in all_edges[:, 1]]

            all_edges_likelihood['from'] = all_edges_likelihood[['first','second']].min(axis=1)
            all_edges_likelihood['to'] = all_edges_likelihood[['first','second']].max(axis=1)


            #Likelihood transformed into standard logistic, acording to equation 8 in the paper
            all_edges_likelihood['likelihood'] = 1/(1+np.exp(-edge_likelihood))
            all_edges_likelihood = all_edges_likelihood[['from','to','likelihood']]


            # Load the modified edges of the graph
            modified_edges = pd.read_csv(datasets_location+dataset_name+'/'+rem_type+'/0'+str(i)+'.txt', header=None, sep=' ', dtype=int)
            modified_edges['from'] = modified_edges[[0,1]].min(axis=1)
            modified_edges['to'] = modified_edges[[0,1]].max(axis=1)

            modified_edges['edge_exists_modified'] = 1
            modified_edges = modified_edges[['from','to','edge_exists_modified']]

            # Combine information and save results
            results = pd.merge(all_edges_likelihood, original_edges, on=['from','to'], how='left')
            results = pd.merge(results, modified_edges, on=['from','to'], how='left')

            results['edge_exists_original'] = results['edge_exists_original'].fillna(0)
            results['edge_exists_modified'] = results['edge_exists_modified'].fillna(0)
            results['dataset'] = dataset_name
            results['modification'] = 'base'

            results.to_csv(datasets_location+dataset_name+'/results/'+rem_type+'_0'+str(i)+'.txt', index=False)# Modification cases