echo " A. Preprocessing Phase :"
echo " Building Triplets from Train data..."
python triplet_model/sc1_build_triplet_train_data.py -i data/
echo " Building Triplets from Test data..."
python triplet_model/sc1_build_triplet_test_data.py -i data/
echo " B. Triplet Model : Using Barcode feature encoding on Full Train dataset of 76 colonies"
python triplet_model/sc1_xgb_trainer.py -i data/
echo " C. Triplet Predictions : Running predictions on test data of 30 colonies using the barcode model"
python triplet_model/sc1_xgb_predictor.py -i data/
echo " D. Tree Reconstruction : Reconstruction tree using the triplet predictions on each test colony"
./tree_reconstruction/ctree -c data/sc1/sc1_xgb_triplet_label_test.out 
echo " E. Scoring Trees"
python triplet_model/sc1_score.py -f tree_reconstruction/trees_submission.txt -g data/gold_standard/Goldstandard_SC1.txt -r scores.txt -p utils/TreeCmp/
