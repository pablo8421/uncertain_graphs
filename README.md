# Working with Uncertain graphs

## Graph modification tool
The graph modification tool can be used by importing the functions found in the script. This are their signature.

```python
def remove_edges(graph: Graph, strategy:str = 'random', order:str = 'desc', pct:float = None, k:float = 1, deep_copy:bool = False, seed:int=None)
```
Where the posible values for strategy are:
* random
* [betweenness](https://en.wikipedia.org/wiki/Betweenness_centrality)

The posible values for order are 'asc' or 'desc', where desc means to priorize for a high value of the metric, and asc to priorize lower values of the metric. Unused with random.

pct and k are to define how many edges to remove. if pct is None, the value of k is used. Otherwise, pct is a number between 0 and 1 that indicates which percentage of the edges to remove. 

deep_copy is a boolean flag that indicates if a deep copy of the graph si to be done before modifying it. And seed mantains results between different runs for the randomness.

```python
def remove_vertices(graph: Graph, strategy:str = 'random', order:str = 'desc', pct:float = None, k:float = 1, deep_copy:bool = False, seed:int=None)
```
Where the posible values for strategy are:
* random
* [degree](https://en.wikipedia.org/wiki/Degree_(graph_theory))
* [betweenness](https://en.wikipedia.org/wiki/Betweenness_centrality)
* [closeness](https://en.wikipedia.org/wiki/Closeness_centrality)
* [clustering_coefficient](https://en.wikipedia.org/wiki/Clustering_coefficient)

An example of its use can be seen in the scripts for the experiments [here](Experiments/modify_datasets.py)

## Experiments

The experiments can be found in the Experiments folder. The most important file here is the [run_experiments.sh](Experiments/run_experiments.sh) script that coordinates the different scripts that are run in order to generate the results. A flag can be set for each script to run or not at the beginning of the bash file, as well as a the list of datasets can be modified on it. The scripts are:

### Modification of graphs

* DATASETS: Which given the list of datasets, runs the scripts that generates the graphs with a given percentage of their edges removed. Script can be found [here](Experiments/modify_datasets.py). The list can be modified in the bash script

### Low-Rank Asymmetric Projections Likelihood
These scripts are based on the code found [here](https://github.com/google/asymproj_edge_dnn). The code is not the same, it has been modified to be runnable in python 3.9 instead of 2.7, but other than that it remains the same.
* DATASETS_ARRAYS: Which generates the necessary training files for the Low-Rank Asymmetric function. The script can be found [here](Experiments/asymproj_edge_dnn/create_dataset_arrays.py)
* EDGE_LIKELIHOOD_TRAINING: Which actually trains each of the functions based on the output by the previous script. It can be found [here](Experiments/asymproj_edge_dnn/deep_edge_trainer.py)

### SEAL Likelihood function
This script is based on the code found [here](https://github.com/muhanzhang/SEAL/tree/master/Python). The code is mostly the same, the only modification is the addition of of a variable to control the output folder. The default was only data.
* SEAL_LIKELIHOOD_SPLIT: Which splits the graphs into train/test split in the format the SEAL training requires it. It also ouputs all the posible edges into 200 different files, to avoid a known problem in the evaluation part.The script can be found [here](Experiments/split_dataset_seal.py)
* SEAL_LIKELIHOOD_TRAINING: Which trains the different models using the files generated from the step before. The script can be found [here](Experiments/seal_link_prediction/SEAL/Python/Main.py)
* SEAL_LIKELIHOOD_EVALUATION: Which evaluates the previous function on all of the 200 files generated in the split part. The script is run once for each file, and checks it's output before running it, to avoid reruns. The script can be found [here](Experiments/seal_link_prediction/SEAL/Python/Main.py)

### Evaluate edge likelihood function (for all methods)
* EDGE_LIKELIHOOD_EVAL: From the trained models outputed by the first function, and the files generated by the second function, It generates a file with all the information for all the posible edges in the different graphs. The script can be found [here](Experiments/evaluate_edge_likelihood.py).

### Create graphs from edge likelihood function
* GENERATE_NEW_GRAPHS: Using the results from the previous script, new graphs are generated. The script can be modified to use one, or the other edge likelihood function. It can be found [here](Experiments/generate_new_graphs.py).

### Running community detection on graphs
* COMMUNITY_DETECTION: Runs the different community detection algorithms on the graphs generated before. It can be found [here](Experiments/evaluate_community_detection.py). 


### Datasets format
The datasets require a specific format to be run. Two files must exists inside the datasets folder in a folder with the name of the dataset. 
* The first file must be named as the dataset and followed by _edges.txt. It must contain the pairwise edges, separated by a space.
* The second file must be named as the dataset and followed by a _comm.txt. It must contain the community belonging of each node, separated by a space.
