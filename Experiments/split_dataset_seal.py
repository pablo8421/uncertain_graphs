import os
import sys
import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
	# This is to keep consistent results when modifing the graph
	seed_to_use = 4269

	output_directory = sys.argv[1]
	file_path = sys.argv[2]
	file_name = sys.argv[3]

	dataset = pd.read_csv(file_path, sep=' ', header=None)

	nodes = list(set(np.concatenate((dataset[0], dataset[1]))))
    # Create all posible edges combination
	all_edges = np.array(list(itertools.combinations(nodes, 2)))
	# Split into train/test
	X_train, X_test = train_test_split(dataset, test_size=0.1, random_state=4269)


	X_train.to_csv(output_directory+file_name+'_train.txt', sep=' ', header=None, index=False)
	X_test.to_csv(output_directory + file_name + '_test.txt', sep=' ', header=None, index=False)


	all_edges_splitted = np.array_split(all_edges, 200)
	for index, edges in enumerate(all_edges_splitted):
		np.savetxt(output_directory + file_name +'_all_'+'{0:03d}'.format(index)+'.txt', edges, fmt=['%d', '%d'])