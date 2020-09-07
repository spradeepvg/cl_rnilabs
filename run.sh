#!/bin/bash

encoding="barcode"
run_phase="all"
file_prefix="sc1_xgb_triplet_label_"

if [ $# -eq 2 ]
then
	run_phase=$1
	encoding=$2
fi

output_fname="${file_prefix}_test_${encoding}.out"
file_prefix=$file_prefix$encoding
model_fname="${file_prefix}.model"
scores_fname="sc1_scores_${encoding}.out"

echo $model_fname
echo $file_prefix
echo $output_fname
echo $scores_fname

echo "Running ${run_phase} pipeline components using ${encoding} .."

if [ "$run_phase" = "all" ] || [ "$run_phase" = "prepare" ]
then
	echo " A. Preprocessing Phase :"
	echo " Building Triplets from Train data..."
	python triplet_model/sc1_build_triplet_train_data.py -i data/ 
	echo " Building Triplets from Test data..."
	python triplet_model/sc1_build_triplet_test_data.py -i data/
fi

if [ "$run_phase" = "all" ] || [ "$run_phase" = "build" ]
then
        if [ "$encoding" = "barcode" ]
        then
		echo " B. Triplet Model : Using Barcode feature encoding on Full Train dataset of 76 colonies"
                python triplet_model/sc1_xgb_trainer.py -i data/
		echo " C. Triplet Predictions : Running predictions on test data of 30 colonies using the barcode model"
                python triplet_model/sc1_xgb_predictor.py -i data/
                echo " D. Tree Reconstruction : Reconstruction tree using the triplet predictions on each test colony"
                ./tree_reconstruction/ctree -c data/sc1/sc1_xgb_triplet_label_test.out
                echo " E. Scoring Trees"
                python triplet_model/score_sc1.py -f trees_submission.txt -g data/gold_standard/Goldstandard_SC1.txt -r data/sc1/results/scores.txt -p utils/TreeCmp/
        else
		echo " B. Triplet Model : Using ${encoding} feature encoding on Full Train dataset of 76 colonies"
                python triplet_model/sc1_xgb_trainer.py -i data/ -enc $encoding -m $model_fname -o $file_prefix
		echo " C. Triplet Predictions : Running predictions on test data of 30 colonies using the barcode model"
                python triplet_model/sc1_xgb_predictor.py -i data/ -enc $encoding -m $model_fname -o $output_fname
                echo " D. Tree Reconstruction : Reconstruction tree using the triplet predictions on each test colony"
                ./tree_reconstruction/ctree -c data/sc1/$output_fname
                echo " E. Scoring Trees"
                python triplet_model/score_sc1.py -f trees_submission.txt -g data/gold_standard/Goldstandard_SC1.txt -r data/sc1/results/$scores_fname -p utils/TreeCmp/
        fi


	echo " Cleaning up intermediate files .."
	rm -rf *.nwk treecmp_results.out
fi

echo "Pipeline execution completed ...."
