import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

if __name__ == "__main__":
	# This is to keep consistent results when modifing the graph
	seed_to_use = 4269

	dataset_source = sys.argv[1]
	#likelihood_to_use = 'asim_likelihood'
	likelihood_to_use = 'seal_likelihood'

	results_rows = []
	for file_name in os.listdir(dataset_source+'results/'):
		if file_name == 'base.txt':
			continue
		dataset = pd.read_csv(dataset_source+'results/'+file_name)
		dataset[likelihood_to_use] = dataset[likelihood_to_use].fillna(0)

		# Split into train/test
		X_train, X_test, E_train, E_test, Y_train,Y_test = train_test_split(dataset[likelihood_to_use], dataset['edge_exists_original'], dataset['edge_exists_modified'],
												 test_size=0.3, random_state=4269, stratify=dataset['edge_exists_modified'])

		# 'Train', get best threshold, via f1_score
		precision, recall, thresholds = precision_recall_curve(Y_train, X_train)
		f1_scores = np.divide(2*recall*precision, (recall+precision), out=np.zeros_like(recall+precision), where=(recall+precision!=0))

		# Get the threshold with the best results
		threshold = thresholds[np.argmax(f1_scores)]
		train_f1_score = np.max(f1_scores)
		# Get test best score
		test_f1_score = f1_score((X_test > threshold).astype('float64').values, Y_test)
		# Comparing to the original graph (just the test ones)
		exists_test_f1_score = f1_score((X_test > threshold).astype('float64').values, E_test)
		# Comparing to the original graph (ALL OF THEM!)
		exists_all_f1_score = f1_score(np.concatenate(((X_train > threshold).astype('float64').values, (X_test > threshold).astype('float64').values)), np.concatenate((E_train,E_test)))
		
		# To save the results of all the training
		results_rows.append({'file_name':file_name, likelihood_to_use:threshold, 'train_f1_score':train_f1_score, 'test_f1_score':test_f1_score, 'exists_f1_score':exists_test_f1_score, 'exists_all_f1_score':exists_all_f1_score})
		
		if file_name != 'base.txt':
			# Creating graph only adding new edges, with weight 1 for all
			selected_edges = dataset[(dataset['edge_exists_modified']==1) | ((dataset[likelihood_to_use]>threshold))].copy()
			selected_edges[['from','to']].to_csv(dataset_source+'resulting_graphs/add_edges/'+file_name, sep=' ', header=None, index=False)
			
			# Creating graph adding the edges with weight likelihood
			selected_edges['edge_weight'] = selected_edges.apply(lambda r: 1.0 if r['edge_exists_modified']==1 else r[likelihood_to_use],axis=1)
			selected_edges[['from','to','edge_weight']].to_csv(dataset_source+'resulting_graphs/add_weighted/'+file_name, sep=' ', header=None, index=False)

			# Creating graph with a weight for all of the edges, existing and added
			selected_edges[['from','to',likelihood_to_use]].to_csv(dataset_source+'resulting_graphs/add_all_weighted/'+file_name, sep=' ', header=None, index=False)

		else:
			# Creating graph as the base graph only
			selected_edges = dataset[(dataset['edge_exists_modified']==1)].copy()
			selected_edges[['from','to']].to_csv(dataset_source+'resulting_graphs/add_edges/'+file_name, sep=' ', header=None, index=False)
			
			# Creating graph adding the edges with weight likelihood
			selected_edges['edge_weight'] = selected_edges.apply(lambda r: 1.0 if r['edge_exists_modified']==1 else r[likelihood_to_use],axis=1)
			selected_edges[['from','to','edge_weight']].to_csv(dataset_source+'resulting_graphs/add_weighted/'+file_name, sep=' ', header=None, index=False)

			# Creating graph with a weight for all of the edges, existing and added
			selected_edges[['from','to',likelihood_to_use]].to_csv(dataset_source+'resulting_graphs/add_all_weighted/'+file_name, sep=' ', header=None, index=False)


	threshold_results = pd.DataFrame(results_rows)
	threshold_results.to_csv(dataset_source+'thresholds_scores.csv', index=False)
