#!/bin/bash

# Decision flags
# Generate all of the subgraphs
RUN_DATASETS=0
# Low-Rank Asymmetric Projections Likelihood
RUN_DATASETS_ARRAYS=0
RUN_EDGE_LIKELIHOOD_TRAINING=0

# SEAL Likelihood function
RUN_SEAL_LIKELIHOOD_SPLIT=0
RUN_SEAL_LIKELIHOOD_TRAINING=0
RUN_SEAL_LIKELIHOOD_EVALUATION=0

# Evaluate edge likelihood function (for all methods)
RUN_EDGE_LIKELIHOOD_EVAL=0

# Create graphs from edge likelihood function
RUN_GENERATE_NEW_GRAPHS=0

# Running community detection on graphs
RUN_COMMUNITY_DETECTION=1

# Helper variables
DATASETS_LOCATION='datasets/'
#DATASETS=('dancer_01' 'dancer_03')
DATASETS=('dancer_01')

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

    # 4 Split original graph into the required files to train and evaluate the SEAL model
    if [[ RUN_SEAL_LIKELIHOOD_SPLIT -eq 1 ]]; then
        # 4.1 Create output directory
        out_dir=${DATASETS_LOCATION}${ds}/seal_data

        mkdir -p $out_dir/

        # 4.2 Split generated graphs into train/test for the seal model training

        # 4.2.1 Base case
        #python split_dataset_seal.py $out_dir/ ${DATASETS_LOCATION}${ds}/${ds}_edges.txt base

        # 4.2.2 random
        mkdir -p $out_dir/random/
        for i in {1..9}
        do
            mkdir -p $out_dir/random/0${i}/
            python split_dataset_seal.py $out_dir/random/0${i}/ ${DATASETS_LOCATION}${ds}/random/0${i}.txt random_0${i}
        done

        # 4.2.3 bet asc
        mkdir -p $out_dir/bet_asc/
        for i in {1..9}
        do
            mkdir -p $out_dir/bet_asc/0${i}/
            python split_dataset_seal.py $out_dir/bet_asc/0${i}/ ${DATASETS_LOCATION}${ds}/bet_asc/0${i}.txt bet_asc_0${i}
        done

        # 4.2.4 bet desc
        mkdir -p $out_dir/bet_desc/
        for i in {1..9}
        do
            mkdir -p $out_dir/bet_desc/0${i}/
            python split_dataset_seal.py $out_dir/bet_desc/0${i}/ ${DATASETS_LOCATION}${ds}/bet_desc/0${i}.txt bet_desc_0${i}
        done

    fi

    # 5 Train the SEAL model for each of the generated graphs
    if [[ RUN_SEAL_LIKELIHOOD_TRAINING -eq 1 ]]; then
        out_dir=${DATASETS_LOCATION}${ds}/seal_data

        # 5.1 base
        #python seal_link_prediction/SEAL/Python/Main.py --train-name base_train.txt --test-name base_test.txt --hop 2 --data-directory $out_dir --save-model

        # 5.2 random
        for i in {1..9}
        do
            python seal_link_prediction/SEAL/Python/Main.py --train-name random_0${i}_train.txt --test-name random_0${i}_test.txt --hop 2 --data-directory $out_dir/random/0${i} --save-model
        done

        # 5.3 bet asc
        for i in {1..9}
        do
            python seal_link_prediction/SEAL/Python/Main.py --train-name bet_asc_0${i}_train.txt --test-name bet_asc_0${i}_test.txt --hop 2 --data-directory $out_dir/bet_asc/0${i} --save-model
        done

        # 5.4 bet desc
        for i in {1..9}
        do
            python seal_link_prediction/SEAL/Python/Main.py --train-name bet_desc_0${i}_train.txt --test-name bet_desc_0${i}_test.txt --hop 2 --data-directory $out_dir/bet_desc/0${i} --save-model
        done
    fi

    # 6 Run the SEAL model evaluation for all of the edges
    if [[ RUN_SEAL_LIKELIHOOD_EVALUATION -eq 1 ]]; then
        out_dir=${DATASETS_LOCATION}${ds}/seal_data

        # 6.1 base
        #python seal_link_prediction/SEAL/Python/Main.py --train-name base_train.txt --test-name base_all_edges.txt --hop 2 --data-directory $out_dir --only-predict

        # 6.2 random
        for i in {1..9}
        do
            for f in $out_dir/random/0${i}/*_all_???.txt;
            do                
                pred_file=${f/.txt/_pred.txt}
                name=${f##*/}
                if test -f "$pred_file"; then
                    echo "Skipping $name, since it has already been processed."
                else
                    python seal_link_prediction/SEAL/Python/Main.py --train-name random_0${i}_train.txt --test-name $name --hop 2 --data-directory $out_dir/random/0${i} --only-predict
                fi
            done
        done

        # 6.3 bet asc
        for i in {1..9}
        do
            for f in $out_dir/bet_asc/0${i}/*_all_???.txt;
            do
                pred_file=${f/.txt/_pred.txt}
                name=${f##*/}
                if test -f "$pred_file"; then
                    echo "Skipping $name, since it has already been processed."
                else
                    python seal_link_prediction/SEAL/Python/Main.py --train-name bet_asc_0${i}_train.txt --test-name $name --hop 2 --data-directory $out_dir/bet_asc/0${i} --only-predict
                fi
            done
        done

        # 6.4 bet desc
        for i in {1..9}
        do
            for f in $out_dir/bet_desc/0${i}/*_all_???.txt;
            do
                pred_file=${f/.txt/_pred.txt}
                name=${f##*/}
                if test -f "$pred_file"; then
                    echo "Skipping $name, since it has already been processed."
                else
                    python seal_link_prediction/SEAL/Python/Main.py --train-name bet_desc_0${i}_train.txt --test-name $name --hop 2 --data-directory $out_dir/bet_desc/0${i} --only-predict
                fi
            done
        done

    fi

    # 6 Evaluate edge likelihood for each of the datasets and store results
    if [[ RUN_EDGE_LIKELIHOOD_EVAL -eq 1 ]]; then

        mkdir -p ${DATASETS_LOCATION}${ds}/results/
        seal_dir=${DATASETS_LOCATION}${ds}/seal_data/

        #Evaluate edge likelihood for each trained model
        python evaluate_edge_likelihood.py $ds $DATASETS_LOCATION $seal_dir
    fi

    # 7 Generate new graphs based on the edge likelihood
    if [[ RUN_GENERATE_NEW_GRAPHS -eq 1 ]]; then

        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/
        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/add_edges/
        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/add_weighted/
        mkdir -p ${DATASETS_LOCATION}${ds}/resulting_graphs/add_all_weighted/

        #Evaluate edge likelihood for each trained model
        python generate_new_graphs.py ${DATASETS_LOCATION}${ds}/
    fi

    # 8 Run community detection
    if [[ RUN_COMMUNITY_DETECTION -eq 1 ]]; then

        graphs_folder=${DATASETS_LOCATION}${ds}/resulting_graphs/
        output_folder=${DATASETS_LOCATION}${ds}/
        community_file=${DATASETS_LOCATION}${ds}/${ds}_comm.txt

        #Evaluate edge likelihood for each trained model
        python evaluate_community_detection.py $ds $graphs_folder $output_folder $community_file
    fi

done