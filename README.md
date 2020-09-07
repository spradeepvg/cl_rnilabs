# Allen Institute Cell lineage Reconstruction Challenge for SC1 - Approach
**Contributors** : _Vangala G. Saipradeep, Naveen Sivadasan, Aditya Rao, Thomas Joseph, Rajgopal Srinivasan_

## Overall Approach
Our lineage tree construction approach consists of two main parts:
* Triplet orientation prediction : In this, we build a prediction model to predict the induced topology of any three input cells in the final lineage tree based on their barcoding. We call this as triplet orientation prediction.
* Tree Reconstruction from Triplets : We use these predicted triplet topologies (orientations) and construct the final lineage tree satisfying these predictions in a maximal fashion. This is performed in a divide and conquer fashion. These two steps are detailed below.
## Methods
### Triplet orientation prediction

* For a given triplet of cells say {a, b, c}, we say that _cell 'a' separates first this triplet set in the final tree_ to indicate that in the final tree, there is a subtree that contains both 'b' and 'c' but not 'a'. In other words, 'a' lies outside the subtree rooted at the least common ancestor of 'b' and 'c'.  We use the notation { a*, b, c} to indicate the same. Similarly, { a, b*, c} indicates that 'b' separates first from this triplet set in the final tree. We call the identification of the element that separates first from the given triplet as the triplet orientation prediction problem.
* We build a prediction model that takes are input a triplet (their barcodes) and predicts the element that separates first in that triplet (its triplet orientation). We view this as a classification problem involving three classes {1, 2, 3} where the output predicted class indicates the element in the input triplet that separates first.
* For each tree in the training data, we build training data elements where a training element contains the triplet and its true class. The triplet elements are encoded using their barcodes. In particular, for a triplet {a, b, c}, we compute their pairwise Hamming vectors, denoted as H(a, b), H(b, c) and H(a, c).  In the Hamming vector H(a, b), its component is '0' if the corresponding barcode components are same for both 'a' and 'b', and additionally this barcode component is '0'. The vector component is '1' if the values are same but the corresponding barcode component is non-zero and the vector component is '2' if their corresponding barcode components are different. The output class is a number from {1, 2, 3}.  We include all permutations of a triplet in the training data. 
* We use XGBoost with hyper parameter tuning.
* We run predictions on the test triplet permutations and predict their triplet orientations.
* For a given triplet set, its final orientation prediction and the associated prediction score is computed by considering the predictions given by the model for all different input permutations for the same triplet.

### Tree construction

Once we perform orientation predictions on all possible cell triplets, using their prediction scores, we construct the tree in a recursive fashion as follows. The recursive tree construction has the following three main steps. For ease of understanding, we say that a "leaf node (cell) 'x' separates first from a set of leaf nodes (cells) S in the final tree" to indicate that, in the final tree, there is a subtree that contains all remaining nodes in S except 'x'. That is, 'x' lies outside the subtree rooted at the least common ancestor of the remaining nodes in S.  

* **Pivot selection**:  We scan the triplets and based on their prediction scores, we choose the most suitable triplet as the pivot. The pivot is then used to perform divide-and-conquer where tree constructions are performed recursively on parts of the original input. Let {a*, b, c} denote the final chosen triplet pivot. That is, 'a' separates first from {a, b, c} in the final tree.

* **Input splitting**:  Using the pivot {a*, b, c}, we split the input to create three different subproblems, namely T1, T2 and T3, which are then solved recursively. Both 'b' and 'c' are assigned to T1 and 'a' is assigned to T2. For each of the remaining input cell 'x', we perform the following. We inspect the triplet scores of the triplets involving 'x' and any two of {a, b, c}. Based on these, 'x' is assigned to either of T1, T2 or T3. Informally speaking, based on the relevant triplet scores, 'x' is assigned to T1 if 'a' separates first from {x, a, b, c} in the final tree, and 'x' is assigned to T2 if either 'b' or 'c' separates first from {x, a, b, c} in the final tree. Finally, 'x' is assigned to T3 if 'x' separates first from {x, a, b, c} in the final tree. Let the tree construction outputs also be denoted as T1, T2 and T3.

* **Joining subtrees T1, T2 and T3 to obtain the final tree**: Tree construction is performed separately for T1, T2 and T3. After this, the resulting subtrees T1, T2, T3 are joined to obtained the final tree as follows. T1, and T2 are joined by a new ancestor to obtain an intermediate tree say TT. Joining of TT with T3 is done carefully by descending from the root of T3 downwards iteratively. Let 'y' be the current internal node of T3 which is being inspected. 'y' is initially the root of T3. We consider the triplet scores for triplets composed of a leaf node from TT and one leaf node each from the two subtrees rooted respectively at the two child nodes of 'y'. Based on these, either, a node split is performed at 'y' to join TT to T3, or, the scanning descends to either the left or the right child of 'y' in T3.

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

Apart from this, **run_all.sh** script builds triplets, trains model, runs triplet predictions, performs Tree reconstruction from triplets and scores the reconstructed trees.

## Running experiments with different encoding and custom pipelines
### A. Preprocessing Phase : Build triplets from data (Train/Test)
``` python triplet_model/sc1_build_triplet_train_data.py -i data/ ```
``` python triplet_model/sc1_build_triplet_test_data.py -i data/ ```

### B. Training Triplet models 
#### Default : Using Barcode encodings as features
``` python triplet_model/sc1_xgb_trainer.py -i data/ ```
#### Using Hamming and Both encodings as features
``` python triplet_model/sc1_xgb_trainer.py -i data/ -enc hamming ```
#### Using both Hamming and Barcode encodings as features
``` python triplet_model/sc1_xgb_trainer.py -i data/ -enc hybrid ```

### C. Running Triplet Predictions
#### Default : Using Barcode encodings as features
``` python triplet_model/sc1_xgb_predictor.py -i data/ ```
#### Using Hamming and Both encodings as features
``` python triplet_model/sc1_xgb_predictor.py -i data/ -enc hamming ```
#### Using both Hamming and Barcode encodings as features
``` python triplet_model/sc1_xgb_predictor.py -i data/ -enc hybrid ```

### D. Running Tree Reconstructions
``` ./tree_reconstruction/ctree -c data/sc1/sc1_xgb_triplet_label_test.out ```

### E. Scoring Reconstructed Trees
``` python triplet_model/score_sc1.py -f trees_submission.txt -g data/gold_standard/Goldstandard_SC1.txt -r data/sc1/results/scores.txt -p utils/TreeCmp/ ```
