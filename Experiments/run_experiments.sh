#!/bin/bash

# Decision flags
RUN_DATASETS=0
RUN_DATASETS_ARRAYS=0
RUN_EDGE_LIKELIHOOD_TRAINING=0
RUN_EDGE_LIKELIHOOD_EVAL=0
RUN_GENERATE_NEW_GRAPHS=1
RUN_COMMUNITY_DETECTION=1

# Helper variables
DATASETS_LOCATION='datasets/'
DATASETS=('dancer_01' 'dancer_03')
#DATASETS=('dancer_03')

# Iterate through all of the datasets listed above
for ds in "${DATASETS[@]}"
do
    # 1. Modify datasets to create all the necessary variations
    if [[ RUN_DATASETS -eq 1 ]]; then
        python modify_datasets.py "$ds" $DATASETS_LOCATION
    fi

    # 2. Preprocess datasets to train the edge likelihood function
    if [[ RUN_DATASETS_ARRAYS -eq 1 ]]; then
        # 2.1 Preprocess unmodified graph

        # Variables to train edge likelihood
        input_file=${DATASETS_LOCATION}${ds}/${ds}_edges.txt
        out_dir=${DATASETS_LOCATION}${ds}/base/

        # Output directory
        mkdir -p $out_dir

        # Data preprocessing
        python asymproj_edge_dnn/create_dataset_arrays.py --input $input_file --output_dir $out_dir

        # 2.2 For the randomly modified graph
        for i in {1..9}
        do         
            # Variables to train edge likelihood
            input_file=${DATASETS_LOCATION}${ds}/random/0${i}.txt
            out_dir=${DATASETS_LOCATION}${ds}/random/0${i}/

            # Output directory
            mkdir -p $out_dir

            # Data preprocessing
            python asymproj_edge_dnn/create_dataset_arrays.py --input $input_file --output_dir $out_dir
        done

        # 2.3 For the graph modfied by betweenness ascending
        for i in {1..9}
        do         
            # Variables to train edge likelihood
            input_file=${DATASETS_LOCATION}${ds}/bet_asc/0${i}.txt
            out_dir=${DATASETS_LOCATION}${ds}/bet_asc/0${i}/

            # Output directory
            mkdir -p $out_dir

            # Data preprocessing
            python asymproj_edge_dnn/create_dataset_arrays.py --input $input_file --output_dir $out_dir
        done

        # 2.4 For the graph modfied by betweenness descending
        for i in {1..9}
        do         
            # Variables to train edge likelihood
            input_file=${DATASETS_LOCATION}${ds}/bet_desc/0${i}.txt
            out_dir=${DATASETS_LOCATION}${ds}/bet_desc/0${i}/

            # Output directory
            mkdir -p $out_dir

            # Data preprocessing
            python asymproj_edge_dnn/create_dataset_arrays.py --input $input_file --output_dir $out_dir
        done
    fi

    # 3 Train edge likelihood function for each of the datasets
    if [[ RUN_EDGE_LIKELIHOOD_TRAINING -eq 1 ]]; then
        # 3.1 For the unmodified graph

        # Variables to train edge likelihood
        out_dir=${DATASETS_LOCATION}${ds}/base/

        # Actual training of edge likelihood function
        python asymproj_edge_dnn/deep_edge_trainer.py --dataset_dir $out_dir

        # 3.2 For the randomly modified graph
        for i in {1..9}
        do         
            # Variables to train edge likelihood
            out_dir=${DATASETS_LOCATION}${ds}/random/0${i}/

            # Actual training of edge likelihood function
            python asymproj_edge_dnn/deep_edge_trainer.py --dataset_dir $out_dir
        done

        # 3.3 For the graph modfied by betweenness ascending
        for i in {1..9}
        do         
            # Variables to train edge likelihood
            out_dir=${DATASETS_LOCATION}${ds}/bet_asc/0${i}/

            # Actual training of edge likelihood function
            python asymproj_edge_dnn/deep_edge_trainer.py --dataset_dir $out_dir
        done

        # 3.4 For the graph modfied by betweenness descending
        for i in {1..9}
        do         
            # Variables to train edge likelihood
            out_dir=${DATASETS_LOCATION}${ds}/bet_desc/0${i}/

            # Actual training of edge likelihood function
            python asymproj_edge_dnn/deep_edge_trainer.py --dataset_dir $out_dir
        done
    fi

    # 4 Evaluate edge likelihood for each of the datasets and store results
    if [[ RUN_EDGE_LIKELIHOOD_EVAL -eq 1 ]]; then

        mkdir -p ${DATASETS_LOCATION}${ds}/results/

        #Evaluate edge likelihood for each trained model
        python evaluate_edge_likelihood.py $ds $DATASETS_LOCATION
    fi

    # 5 Generate new graphs based on the edge likelihood
    if [[ RUN_GENERATE_NEW_GRAPHS -eq 1 ]]; then

        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/
        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/add_edges/
        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/add_weighted/
        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/add_all_weighted/

        #Evaluate edge likelihood for each trained model
        python generate_new_graphs.py ${DATASETS_LOCATION}${ds}/
    fi

    # 6 Run community detection
    if [[ RUN_COMMUNITY_DETECTION -eq 1 ]]; then

        graphs_folder=${DATASETS_LOCATION}${ds}/resulting_graphs/
        output_folder=${DATASETS_LOCATION}${ds}/
        community_file=${DATASETS_LOCATION}${ds}/${ds}_comm.txt

        #Evaluate edge likelihood for each trained model
        python evaluate_community_detection.py $ds $graphs_folder $output_folder $community_file
    fi

done


