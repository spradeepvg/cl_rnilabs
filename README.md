# Allen Institute Cell lineage Reconstruction Challenge for Sub-Challenge #1
**Contributors** : _Vangala G. Saipradeep, Naveen Sivadasan, Aditya Rao, Thomas Joseph, Rajgopal Srinivasan_

## Prerequisites
* python 3.7,
* Python packages : xgboost, h5py, pickle, itertools, json, dendropy, subprocess, sklearn, psutil, newick
* gcc 9.3
* JDK 1.8

## Directory structure

The project contains four sub-folders :

* data : The folder contains train and test datasets for sub-challenge 1
* tree_reconstruction : The folder contains cpp code for reconstruction of trees from predicted triplets
* triplet_model : The folder contain python code for build triplets, train various models, Triplet predictions and scoring reconstructed trees.
* utils : Third-party libraries for Tree Comparison [https://github.com/TreeCmp/TreeCmp]

## Running Pipeline with different feature encodings

**run.sh** script builds triplets, trains various models, runs triplet predictions, performs Tree reconstruction from triplets and scores the reconstructed trees. The current version of code provides improved performance on sub-challenge #1 test data compared to our submission scores as shown in Table 1. The improvement in performance is due to minor bug fixes and fine-tuning of XGBoost hyper parameters.

### Full Pipeline :

* Run all steps from **1-5** on default encoding - barcode
``` ./run.sh all ``` 
* For hammingcode encoding
``` ./run.sh all hammingcode ```
* For both barcode+hammingcode encoding
``` ./run.sh all hybrid  ``` 

### Pre-Processing Phase :

1. Builds triplets from Train and Test datasets
* Run only preprocessing phase
``` ./run.sh prepare ```

### Processing Phase :

2. Build Triplet models using the specified encodings as features 
3. Run Triplet Predictions using the specified encoding model
4. Reconstruct Lineage Tree using the triplet predictions
5. Score the Reconstructed Trees

* Run all steps from **2-5** on default encoding - barcode
``` ./run.sh build ``` 
* For hammingcode encoding
``` ./run.sh build hammingcode  ```
* For both barcode+hammingcode encoding
``` ./run.sh build hybrid  ``` 

## Results
### Table 1 : RF and Triplet averages over different encodings

Code       | RF_average | Triplet_average
-----------|--------- | -----------
Submission | 0.6643   | 0.6342
Current    | **0.6060**     | **0.5745**
