
#include "helper.h"

using namespace std;

//triplet prediction

struct pred_rec{

	int pred_class = 0; //class corresponding to the max score..
	double score = 0; //max score
	int nodes[3]={0,0,0}; // the three elements in the order in which they were input to the predictor.
	double all_scores[3] = {0, 0, 0}; //Scores for predictions of the nodes in the same order as in nodes array.
	int true_class=0;
	string labels[3]; //any labels such as barcodes..
};



struct node_rec{ //tree node;
	node_rec * left = NULL;
	node_rec* right = NULL;
	int  id = -1; 
	string label; //things like barcode etc..
};


class CTree{

	public:
	unordered_map<string, pred_rec> pred_map;
	unordered_map<int, string> nodemap;
	int dreamID = -1;

	unordered_map<string, double> pred_lookup; //To lookup prediction scores for a given triplet configuration. pred_lookup["a#b#c"] (always b < c) gives the average score for triplet configuration (a, (b, c)) [which is same as (a, (c, b)) ]

	bool best_pivot_maj = false; //previous version.

	public:
	void print_triplet(pred_rec & t);
	bool load_rf_triplet_predictions(ifstream & file);
	void get_t_pred(int * elems, pred_rec & rec);	
	void  get_best_t_pred_score(int * elems, pred_rec & rec);	 //best prediction involving the triplets..based on the highest scoring triplet score among the 6 permutations
	void  get_best_t_pred_maj(int * elems, pred_rec & rec);	 //best prediction involving the triplets.. based on the majority support from the 6 possible permutations
	void  get_best_t_pred(int * elems, pred_rec & rec);	 //best prediction involving the triplets..
	node_rec * get_left_most_child(node_rec *tree);
	void get_all_leaf_vals(node_rec * tree, vector<int> &vals); //recursive
	void print_tree_newick(node_rec * tree, stringstream & ss, bool print_label=false);
	node_rec * construct_tree_rec(vector<pred_rec> triplets, vector<int> nodes, int depth); //returns a pointer to the root of the tree. Recursive..
	node_rec * construct_tree(); //returns a pointer to the root of the tree. Recursive..
	void gen_triplets(node_rec * T, vector<pred_rec> &lets, vector<node_rec> & nodes); //recursive..
	node_rec * parse_tree(string & trstr, int & idx);	 //assuming binary tree..
	string normalize_tree(node_rec * T); //for exact tree maching invariant to left/right flips..
	node_rec * old_stitch_trees_rec(node_rec * TT, node_rec * T3, vector<pred_rec> & final_triplets, int & rt_tree_partition, unordered_map <int, int> & partition);
	node_rec * stitch_trees_rec(node_rec * TT, node_rec * T3, vector<pred_rec> & final_triplets, int & rt_tree_partition, unordered_map <int, int> & partition);

	string get_triplet_set_sig(pred_rec& rec);
	void get_best_pivot_maj(vector<pred_rec> & triplets, pred_rec &res);
	void old_get_best_pivot(vector<pred_rec> & triplets, pred_rec &res);
	void get_best_pivot(vector<pred_rec> & triplets, pred_rec &res, vector<int>&nodes);
	string get_lookup_key(int * nodes);
	double get_separation_score(int n1, int n2, vector<int>&nodes); //return \sum [1-prediction((z, (n1, n2))]  for all z not= n1 or n2.
	void get_best_pivot_multi_score(vector<pred_rec> & triplets, pred_rec &res, vector<int>& nodes); //this is to identify a pivot that achives the root level parition of the remaining nodes.. That is, T3 is empty..



};

void CTree::print_triplet(pred_rec & t){
	cout << t.nodes[0] << ", " << t.nodes[1] << ", " << t.nodes[2] << ": " << (t.pred_class) << " : " << (t.true_class) << " (*) : " << t.score << endl;
}

bool CTree::load_rf_triplet_predictions(ifstream & tripf){

	string fline;
	streampos oldpos = tripf.tellg();
	//cout << "Starting at " << oldpos << " " << dreamID << endl;
	if(dreamID < 0 ) getline(tripf, fline); //skip the first line..	
	bool done = true;
	int oldDream = dreamID;
	bool first = true;
	while(getline(tripf, fline)){
		//cout << fline << endl;
		//cout << "Line : " << fline << endl;
		vector<string> line_arr = split(fline, ',');
		pred_rec rec;
	
		int drid = atoi(line_arr[0].c_str());
		if((!first) && (drid != dreamID)){ //done this part,, reset and exit..
			//cout << "Resetting ... to " << oldpos <<  endl;
			tripf.clear();
			tripf.seekg(oldpos);
			//cout << "Reset ... to " << tripf.tellg() <<  endl;
			done = false; //more to go..
			break;
		} 
		first = false;
		dreamID = drid;
		
		
		rec.true_class = atoi(line_arr[5].c_str())  - 1; //only for training data, else ignore this value.

		int pred_index = 5; //set it 6 for train data.. Default for test data
		if(line_arr.size() == 11){
			 pred_index = 6;
		}

		rec.pred_class = atoi(line_arr[pred_index].c_str())  - 1;
		


		//### strictly for DEBUG
		//rec.pred_class = rec.true_class; //This is for debugging and verification..

		rec.score = atof(line_arr[pred_index + 1].c_str());
		rec.nodes[0] = atoi(line_arr[2].c_str());
		rec.nodes[1] = atoi(line_arr[3].c_str());
		rec.nodes[2] = atoi(line_arr[4].c_str());

		//cout << "test" << endl;
		rec.all_scores[0] = atof(line_arr[pred_index + 2].c_str());
		rec.all_scores[1] = atof(line_arr[pred_index + 3].c_str());
		rec.all_scores[2] = atof(line_arr[pred_index + 4].c_str());


		string key = line_arr[2] + "," + line_arr[3] + "," + line_arr[4];
		pred_map[key] = rec;
		//print_triplet(rec);

		//also load pred_lookup...
		int key_nodes[3];

		key_nodes[0] = rec.nodes[0];
		key_nodes[1] = rec.nodes[1];
		key_nodes[2] = rec.nodes[2];
		string look_key_0 = get_lookup_key(key_nodes);
		
		key_nodes[0] = rec.nodes[1];
		key_nodes[1] = rec.nodes[0];
		key_nodes[2] = rec.nodes[2];
		string look_key_1 = get_lookup_key(key_nodes);

		key_nodes[0] = rec.nodes[2];
		key_nodes[1] = rec.nodes[1];
		key_nodes[2] = rec.nodes[0];
		string look_key_2 = get_lookup_key(key_nodes);


		pred_lookup[look_key_0] = (pred_lookup[look_key_0] + rec.all_scores[0]);
		pred_lookup[look_key_1] = (pred_lookup[look_key_1] + rec.all_scores[1]);
		pred_lookup[look_key_2] = (pred_lookup[look_key_2] + rec.all_scores[2]);


		vector<string> node_arr = split(line_arr[1], '_');


		nodemap[rec.nodes[0]] = node_arr[0];
		nodemap[rec.nodes[1]] = node_arr[1];
		nodemap[rec.nodes[2]] = node_arr[2];
		//nodeset.insert(rec.nodes[1]);
		//nodeset.insert(rec.nodes[2]);

		oldpos = tripf.tellg();
		//cout << "Pos " << oldpos << endl;
	}

	return done;
}

string CTree::get_lookup_key(int * nodes){

	return to_string(nodes[0]) + "#" + ((nodes[1] < nodes[2])? (to_string(nodes[1]) + "#" + to_string(nodes[2]) ) : ( to_string(nodes[2]) + "#" + to_string(nodes[1]))); 
}

void CTree::get_t_pred(int * elems, pred_rec & rec){	
	string key;
	key.append(to_string(elems[0])).append(",").append(to_string(elems[1])).append(",").append(to_string(elems[2]));
	//cout << "Pred key : " << key << endl;
	rec = pred_map[key];
	//cout << "*Return  : "<< endl;
	//print_triplet(rec);
}



void  CTree::get_best_t_pred_score(int * elems, pred_rec & rec){	 //best prediction involving the triplets..based on the highest scoring triplet score among the 6 permutations

	rec.score = 0;
	//cout << "Best pred for : " << elems[0] << " " << elems[1] << " " << elems[2] << endl;

	int idx[][6] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
	for(int i =0; i < 6; ++i){
		int inp[3]; //a permutation..
		inp[0] = elems[idx[i][0]];
		inp[1] = elems[idx[i][1]];
		inp[2] = elems[idx[i][2]];	
		pred_rec mrec;
		get_t_pred( inp, mrec );
		//cout << "GOT : " << i << endl;
		//print_triplet(mrec);
		if(mrec.score > rec.score){
			rec = mrec;
		}
	}
	//cout << "Return  : "<< endl;
	//print_triplet(rec);
}

void  CTree::get_best_t_pred_maj(int * elems, pred_rec & rec){	 //best prediction involving the triplets.. based on the majority support from the 6 possible permutations

	rec.score = 0;
	//cout << "Best pred for : " << elems[0] << " " << elems[1] << " " << elems[2] << endl;

	int idx[][6] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
	double support[3] = {0, 0, 0};
	int nsupp[3] = {0, 0, 0};
	int inp[3]; //a permutation..
	for(int i =0; i < 6; ++i){
		inp[0] = elems[idx[i][0]];
		inp[1] = elems[idx[i][1]];
		inp[2] = elems[idx[i][2]];	
		pred_rec mrec;
		get_t_pred( inp, mrec );
		//cout << "Cand. pred " << endl;
		//print_triplet(mrec);

		if(mrec.nodes[mrec.pred_class] == elems[0]){
			support[0] += mrec.score;
			++nsupp[0];
		}else if(mrec.nodes[mrec.pred_class] == elems[1]){
			support[1] += mrec.score;
			++nsupp[1];
		}else {
			support[2] += mrec.score;
			++nsupp[2];
		}
	}
	int max = 0;
	//for(int i=0; i < 3; ++i) cout << "For " << elems[i] << " supp " << support[i] << " nsupp " << nsupp[i] << endl;
	if(support[1] > support[max]) max = 1;
	if(support[2] > support[max]) max = 2;	
	//cout << "Max supp : " << max <<endl;
	rec.nodes[0] = elems[0];
	rec.nodes[1] = elems[1];
	rec.nodes[2] = elems[2];
	rec.pred_class = max;
	rec.score = (support[max] / (double)nsupp[max]); //avg support score..
	//get the max support class...
	//print_triplet(rec);
}

void  CTree::get_best_t_pred(int * elems, pred_rec & rec){	 //best prediction involving the triplets..
	//get_best_t_pred_score(elems, rec);
	get_best_t_pred_maj(elems, rec);
}

node_rec * CTree::get_left_most_child(node_rec *tree){

	node_rec * n = tree; // assumed to be non NULL
	while(n ->left != NULL){
		n = n->left;
	}
	return n;
}

void CTree::get_all_leaf_vals(node_rec * tree, vector<int> & vals){ //recursive

	if(tree == NULL) return;

	if((tree->left == NULL) && (tree->right == NULL)) {
		vals.push_back(tree->id);
		return;
	}
		
	get_all_leaf_vals(tree->left, vals);
	get_all_leaf_vals(tree->right, vals);
}

void CTree::print_tree_newick(node_rec * tree, stringstream & ss, bool print_label){
	
	if(tree == NULL) {
		ss << "()" << endl;
		return;
	}

	if((tree->left == NULL) && (tree->right == NULL)) {
		if(print_label){
			ss << tree->id << "_" << nodemap[tree->id] ;
		}else{
			ss << tree->id ;
		}
		return;
	}
	ss << "(";
	print_tree_newick(tree->left, ss, print_label);
	ss << ",";
	print_tree_newick(tree->right, ss, print_label);
	ss << ")";
}


node_rec * CTree::old_stitch_trees_rec(node_rec * TT, node_rec * T3, vector<pred_rec> & final_triplets, int & rt_tree_partition, unordered_map < int, int> & partition){


	if(T3 == NULL){ 
		return TT;
	} 

	if(T3->left == NULL) {//leaf node. also T3.right == NULL
		node_rec * T = new node_rec();
		T->left = TT;	
		T->right = T3;
		return T;
	}

	vector<int> rt_nodes;
	get_all_leaf_vals(T3->right, rt_nodes); //get the rt subtree of T3.

	for(auto yy: rt_nodes){
		partition[yy] = 4; //rt ones are now in a new partition..
	}

	//get the best triplet from final_triplets having one element in T3 and another in T4.

	pred_rec best_pred;
	int best_tt_idx = 0;
	for(auto x: final_triplets){
		int p1 = partition[x.nodes[0]];
		int p2 = partition[x.nodes[1]];
		int p3 = partition[x.nodes[2]];

		
		//clearly exactly two of them are in T3 U T4.

		int tt_idx = 0;
		if(p1 < 3) { 
			tt_idx = 0;
		}else if(p2 < 3){
			tt_idx = 1;
		}else {
			tt_idx = 2;
		}
	
		// Look for a triplet where exactly one if it is in T4. These triplet are such that exactly one is in T3.left, one in T4 and one in TT.
		int ncount =0;
		if(p1 == 4) ++ncount;
		if(p2 == 4) ++ncount;
		if(p3 == 4) ++ncount;
		if(ncount != 1) continue;

		if(x.score > best_pred.score) {
			best_pred = x;
			best_tt_idx = tt_idx;
		}
	} 	

	//Now combine..
	if(best_pred.pred_class = best_tt_idx){//triplet winner is from tree TT.
		node_rec * T = new node_rec();
		T->left = TT;	
		T->right = T3;
		return T;
	}else{
		node_rec * lp = get_left_most_child(T3);
		node_rec * nnode = new node_rec();
		nnode->id = lp->id;
		lp->right = nnode;
		lp->left = TT;
		return T3;

	}
}

node_rec * CTree::stitch_trees_rec(node_rec * TT, node_rec * T3, vector<pred_rec> & final_triplets, int & rt_tree_partition,  unordered_map < int, int > & partition ){ //recursive greedy...


	if(T3 == NULL){ 
		return TT;
	} 

	if(T3->left == NULL) {//leaf node. also T3.right == NULL
		node_rec * T = new node_rec();
		T->left = TT;	
		T->right = T3;
		return T;
	}

	vector<int> rt_nodes;
	get_all_leaf_vals(T3->right, rt_nodes); //get the rt subtree of T3.

	for(auto yy: rt_nodes){
		//cout << "rt part : " << yy << endl;
		partition[yy] = rt_tree_partition; //rt ones are now in a new partition.. Initially, this is 4.
	}

	//cout << "See partition : " <<endl;
	//for(auto uu: partition) cout << uu.first << " ::: " << uu.second << endl;
	//get the best triplet from final_triplets having one element in T3 and another in T4.

	pred_rec best_pred;
	vector<pred_rec> new_final_rt;
	vector<pred_rec> new_final_left;
	int best_parts[3];

	for(auto x: final_triplets){
		//cout << "***" << endl;
		//print_triplet(best_pred);
		int parts[3];
		parts[0] = partition[x.nodes[0]];
		parts[1] = partition[x.nodes[1]];
		parts[2] = partition[x.nodes[2]];

		
		//clearly exactly two of them are in T3.

		// Look for a triplet where exactly one if it is in T4. These triplet are such that exactly one is in T3.left, one in T4 and one in TT.
		int nrcount =0;
		if(parts[0] == rt_tree_partition) ++nrcount;
		if(parts[1] == rt_tree_partition) ++nrcount;
		if(parts[2] == rt_tree_partition) ++nrcount;
		if(nrcount != 1) {
			if(nrcount == 2) {
				new_final_rt.push_back(x);
			}else{ //nrcount == 0 (fully left..
				new_final_left.push_back(x);
			}
			continue;
		}
		//one rt and one left... in T3

		//new_final.push_back(x);

		//cout << "x ";
		//print_triplet(x);
		if(x.score > best_pred.score) {
			best_pred = x;
			//print_triplet(best_pred);
			best_parts[0] = parts[0];
			best_parts[1] = parts[1];
			best_parts[2] = parts[2];
		}
	} 	

	//cout << "b0 " << best_parts[0] << " b1 " << best_parts[1] << " b2 " << best_parts[2] << " " << best_parts[best_pred.pred_class] << " " << rt_tree_partition << endl;

	//Now combine..
	if(best_parts[best_pred.pred_class] < 3){//triplet winner is from tree TT.
		node_rec * T = new node_rec();
		T->left = TT;	
		T->right = T3;
		return T;
	}else{//recurse...
		final_triplets.clear();

		node_rec * nT3; 
		if(best_parts[best_pred.pred_class] == rt_tree_partition){ //stitch TT inside T3->left
			nT3 = T3->left;
			//cout << "Stitching in left... " << endl;
			//stringstream ss;
			//print_tree_newick(nT3, ss);
			//cout << ss.str() << endl;
			//cout << "Pred: " << endl;
			//print_triplet(best_pred);
			++rt_tree_partition;
			final_triplets.insert(final_triplets.end(), new_final_left.begin(), new_final_left.end());
			node_rec * T = stitch_trees_rec(TT, nT3, final_triplets, rt_tree_partition, partition); //recursive greedy...
			T3->left = T;
		}else{ //stitch TT inside T3->right..
			nT3 = T3->right;
			//cout << "Stitching in right... " << endl;
			//stringstream ss;
			//print_tree_newick(nT3, ss);
			//cout << ss.str() << endl;
			//cout << "Pred: " << endl;
			//print_triplet(best_pred);
			++rt_tree_partition;
			final_triplets.insert(final_triplets.end(), new_final_rt.begin(), new_final_rt.end());
			node_rec * T = stitch_trees_rec(TT, nT3, final_triplets, rt_tree_partition, partition); //recursive greedy...
			T3->right = T;
		}
		
		return T3;
	}
}


void CTree::old_get_best_pivot(vector<pred_rec> & triplets, pred_rec &res){
	res.score = 0;
	for(auto x : triplets){
		if(x.score > res.score) res = x;
	}

}

string CTree::get_triplet_set_sig(pred_rec& rec){
	
	int nodes[3];
	nodes[0] = rec.nodes[0];
	nodes[1] = rec.nodes[1];
	nodes[2] = rec.nodes[2];
	//sort them..
	sort(nodes, nodes + 3);

	return to_string(nodes[0]) + "#" + to_string(nodes[1]) + "#" + to_string(nodes[2]);

}
void CTree::get_best_pivot_maj(vector<pred_rec> & triplets, pred_rec &res){
	res.score = 0;
	unordered_set <string> trip_sets;

	/*
	pred_rec test;
	int test_nodes[3] = {3,4,6};
	cout << "SEEEE: 0" << endl;
	get_best_t_pred(test_nodes, test);
	cout << "SEEEE: " << endl;
	print_triplet(test);
	*/

	for(auto x : triplets){
		string sig = get_triplet_set_sig(x);
		if(trip_sets.find(sig) != trip_sets.end()) continue;
		pred_rec mrec;
		get_best_t_pred(x.nodes, mrec);
		if(mrec.score > res.score) res = mrec;
		trip_sets.insert(sig);
		//cout << "Cand split : " << endl;
		//print_triplet(mrec);
	}

}

void CTree::get_best_pivot_multi_score(vector<pred_rec> & triplets, pred_rec &res, vector<int>&nodes){ //this is to identify a pivot that achives the root level parition of the remaining nodes.. That is, T3 is empty..
	res.score = 0;
	unordered_set <string> trip_sets;
	double max_sep_score1 = 0, max_sep_score2 = 0;
	double max_pred_score = 0;

	/*
	pred_rec test;
	int test_nodes[3] = {3,4,6};
	get_best_t_pred(test_nodes, test);
	cout << "SEEEE: " << endl;
	print_triplet(test);
	*/

	for(auto x : triplets){
		string sig = get_triplet_set_sig(x);
		if(trip_sets.find(sig) != trip_sets.end()) continue;
		pred_rec mrec;
		get_best_t_pred(x.nodes, mrec);
		int z = mrec.pred_class;
		int other =  (z + 1) %3;
		double sep_score1 = get_separation_score(mrec.nodes[z], mrec.nodes[other], nodes);
		other =  (z + 2) %3;
		double sep_score2 = get_separation_score(mrec.nodes[z], mrec.nodes[other], nodes);
		if((mrec.score + sep_score1 + sep_score2) > (max_pred_score + max_sep_score1 + max_sep_score2)){ //could be other functions/norms over these three quantities.. need to try out in future...
			res = mrec;
			max_sep_score1 = sep_score1;
			max_sep_score2 = sep_score2;
			max_pred_score = mrec.score;
		}
		trip_sets.insert(sig);
		//cout << "Cand pivot : " << endl;
		//print_triplet(mrec);
		//cout << mrec.score << " " << sep_score1 << " " << sep_score2 << endl;
	}
	//cout << "Max pivot : " << endl;
	//print_triplet(res);
	//cout << res.score << " " << max_sep_score1 << " " << max_sep_score2 << endl;

}

void CTree::get_best_pivot(vector<pred_rec> & triplets, pred_rec &res, vector<int>& nodes){
	//old_get_best_pivot(triplets, res);
	if(best_pivot_maj){
		get_best_pivot_maj(triplets, res);
	}else{
		get_best_pivot_multi_score(triplets, res, nodes);
	}

}

double CTree::get_separation_score(int n1, int n2, vector<int>& nodes){ //return \sum [1-prediction((z, (n1, n2))]  for all z not= n1 or n2.

	double score = 0;
	for(auto z: nodes){

		if((z == n1) || (z == n2)) continue;
		int key_nodes[3];
		key_nodes[0] = z;
		key_nodes[1] = n1;
		key_nodes[2] = n2;

		string key = get_lookup_key(key_nodes); //order of n1 and n2 does not matter...

		//cout << "Look up key : val " << key << " : " << pred_lookup[key] << endl; 
		score += (6-pred_lookup[key]); //6 possible combinations of each triplet set is considered.

	}

	return score;
}


node_rec * CTree::construct_tree_rec(vector<pred_rec> triplets, vector<int> nodes, int depth){ //returns a pointer to the root of the tree. Recursive..

	bool debug = false;

	if(debug){
		cout << "\n>>>Calling construct with " << endl;
		cout << "Depth : " << depth << endl;
		cout << "Triplets ... " << endl;
		for(auto t : triplets) print_triplet(t);
		cout << "Nodes ... " << endl;
		for(auto t : nodes) cout << t << ", " << endl;
	}

	//Current implementation is a greedy recursive alg.

	//base case
	if(nodes.size() == 0) {
		return NULL;	
	}else if( nodes.size() == 1 ){
		node_rec * node = new node_rec();
		node->id = nodes[0];
		if(debug){
			stringstream ss;
			cout << "Return tree (single): " << endl;
			print_tree_newick(node, ss);
			cout << ss.str() << endl;
		}
		return node;

	}else if( nodes.size() == 2) {
		node_rec * root = new node_rec();

		node_rec * node = new node_rec();
		node->id = nodes[0];
		root->left = node;

		node = new node_rec();
		node->id = nodes[1];
		root->right = node;

		if(debug){
			stringstream ss;
			cout << "Return tree (two): " << endl;
			print_tree_newick(root, ss);
			cout << ss.str() << endl;
		}
		return root;
	}

	//at least 3 nodes and hence the triplets vector is non-empty

	pred_rec max_rec; //total sorting/ heap complexity..
	get_best_pivot(triplets, max_rec, nodes);


	unordered_map < int, int > partition;
	vector<int> T1_nodes, T2_nodes, T3_nodes;
	vector<pred_rec> T1_pred, T2_pred, T3_pred;

	if(debug){
		cout << "\n***Split triplet " << endl;
		print_triplet(max_rec);
		cout << endl;
	}

	int split_nodes_1[3], split_nodes_2[3];
	if(max_rec.pred_class == 0){
		split_nodes_1[0] = max_rec.nodes[0];
		split_nodes_2[0] = max_rec.nodes[0];

		split_nodes_1[1] = max_rec.nodes[1];
		split_nodes_2[1] = max_rec.nodes[2];


		partition[max_rec.nodes[0]] = 2;
		partition[max_rec.nodes[1]] = 1;
		partition[max_rec.nodes[2]] = 1;

		T1_nodes.push_back(max_rec.nodes[1]);
		T1_nodes.push_back(max_rec.nodes[2]);
		T2_nodes.push_back(max_rec.nodes[0]);
		
	}else if(max_rec.pred_class == 1){
		split_nodes_1[0] = max_rec.nodes[1];
		split_nodes_2[0] = max_rec.nodes[1];

		split_nodes_1[1] = max_rec.nodes[0];
		split_nodes_2[1] = max_rec.nodes[2];

		partition[max_rec.nodes[1]] = 2;
		partition[max_rec.nodes[0]] = 1;
		partition[max_rec.nodes[2]] = 1;


		T1_nodes.push_back(max_rec.nodes[0]);
		T1_nodes.push_back(max_rec.nodes[2]);
		T2_nodes.push_back(max_rec.nodes[1]);

	}else{	
		split_nodes_1[0] = max_rec.nodes[2];
		split_nodes_2[0] = max_rec.nodes[2];

		split_nodes_1[1] = max_rec.nodes[0];
		split_nodes_2[1] = max_rec.nodes[1];

		partition[max_rec.nodes[2]] = 2;
		partition[max_rec.nodes[1]] = 1;
		partition[max_rec.nodes[0]] = 1;


		T1_nodes.push_back(max_rec.nodes[0]);
		T1_nodes.push_back(max_rec.nodes[1]);
		T2_nodes.push_back(max_rec.nodes[2]);

	}


	
	for(auto y : nodes){
		if(partition[y] != 0) {//already processed as split vertices..
			continue;
		}
		split_nodes_1[2] = y;
		split_nodes_2[2] = y;

		pred_rec r1, r2;
		int * split_nodes = split_nodes_1;

		get_best_t_pred(split_nodes_1, r1);
		get_best_t_pred(split_nodes_2, r2);

		if(r2.score > r1.score) {
			r1 = r2;
			split_nodes = split_nodes_2;
		}

		//cout << "Best partition triplet for " << y << endl;
		//print_triplet(r1);
		//cout << endl;

		int mclass = 0;
		int el = r1.nodes[r1.pred_class];
		if(el == split_nodes[0]){
			mclass = 0;
		}else if(el == split_nodes[1]){
			mclass = 1;
		}else{
			mclass = 2;
		}

		if(mclass == 0){
			T1_nodes.push_back(y);
			partition[y] = 1;
			
		}else if(mclass == 1){
			T2_nodes.push_back(y);
			partition[y] = 2;
		}else {	
			T3_nodes.push_back(y);
			partition[y] = 3;
		}
		
	}
	

	vector<pred_rec> final_triplets;

	for(auto x : triplets){
		int p1 = partition[x.nodes[0]];
		int p2 = partition[x.nodes[1]];
		int p3 = partition[x.nodes[2]];
	
		if((p1 != p2) || (p1 != p3)){ //if they are all not the same...
			int ncount = 0;
			if(p1 == 3) ++ncount;
			if(p2 == 3) ++ncount;
			if(p3 == 3) ++ncount;
	
			if(ncount == 2) { //exactly two of them are in T3
				final_triplets.push_back(x);
			}

			continue;
		}

		//if( p != partition[x.nodes[1]]) continue; //split triplets are removed.
		//if( p != partition[x.nodes[2]]) continue;

		if(p1 == 1){
			T1_pred.push_back(x);
		}else if(p1 == 2){
			T2_pred.push_back(x);
		}else {
			T3_pred.push_back(x);
		}
	
	}

	//input split done.. recurse..


	node_rec * T1, * T2, * T3;

	if(debug) cout << "Calling construct on T1 " << endl;
	T1 = construct_tree_rec(T1_pred, T1_nodes, depth+1);
	if(debug) cout << "Calling construct on T2 " << endl;
	T2 = construct_tree_rec(T2_pred, T2_nodes, depth+1);
	if(debug) cout << "Calling construct on T3 " << endl;
	T3 = construct_tree_rec(T3_pred, T3_nodes, depth+1);

	//Now combine these..	

	//Tree is assumed to be full (all internal nodes have two children)
	
	node_rec * TT = new node_rec();
	TT->left = T1;
	TT->right = T2;


	int rt_partition = 4;
	//return old_stitch_trees_rec(TT, T3, final_triplets, rt_partition, partition);

	if(debug){
		stringstream ss;
		cout << "For stitching tree (inp TT): " << endl;
		print_tree_newick(TT, ss);
		cout << ss.str() << endl;

		cout << "For stitching tree (inp T3): "  << endl;
		stringstream ss2;
		print_tree_newick(T3, ss2);
		cout << ss2.str() << endl;
	}


	node_rec * nT =  stitch_trees_rec(TT, T3, final_triplets, rt_partition, partition);

	
	
	if(debug){
		stringstream ss;
		cout << "Return tree (stitched): " << endl;
		print_tree_newick(nT, ss);
		cout << ss.str() << endl;
	}

	return nT;


	//T3 has two or more nodes.. Combine in a greedy fashion

	//recursive stitch operation...


	/*
		int inp[3];
		inp[0] = T1_nodes[0]; // this could be any nodes from T1 and T2 (perhaps the ones that give the best score ones... later)

		node_rec * lp = get_left_most_child(T3);
		inp[1] = lp->id;
		inp[2] = get_left_most_child(T3.right)->id;

		pred_rec mrec;
		get_best_t_pred(inp, mrec);
		if(mrec.pred_class == 0) {

			node_rec * T = new node_rec();
			T.left = TT;	
			T.right = T3;
			return T;
		}else{

			node_rec * nnode = new node_rec();
			nnode.id = lp->id;
			lp->right = nnode;
			lp->left = TT;
			return T3;
		}
	*/
	
}

node_rec * CTree::construct_tree(){ //returns a pointer to the root of the tree. Recursive..

	vector<pred_rec> inp_trip;

	for(auto x : pred_map){
		inp_trip.push_back(x.second);
	} 

	vector<int> inp_nodes;
	for(auto mr : nodemap) {
		inp_nodes.push_back(mr.first);
	}

	return construct_tree_rec(inp_trip, inp_nodes,1 );

}

void CTree::gen_triplets(node_rec * T, vector<pred_rec> &lets, vector<node_rec> & nodes){ //recursive..

	if(T == NULL) return;
	if((T->left == NULL) || (T->right == NULL)){
		nodes.push_back((*T));	 //leaf nodes..
		return;
	}
	vector<node_rec> lnodes;
	vector<node_rec> rnodes;
	gen_triplets(T->left, lets, lnodes);
	gen_triplets(T->right, lets, rnodes);

	//now include the additional ones..

	for(auto x: lnodes){
		for(auto y: lnodes){
			if(y.id == x.id) continue;
			for(auto z: rnodes){ //(x,y) and (y,x) will happen separately..
				pred_rec rec;
				rec.score = 1.0;
				rec.nodes[0] = x.id;
				rec.nodes[1] = y.id;
				rec.nodes[2] = z.id;
				rec.labels[0] = x.label;
				rec.labels[1] = y.label;
				rec.labels[2] = z.label;
				rec.true_class = 2;
				lets.push_back(rec);
				//cout << "Insert rec for z = " << z << endl;
				//print_triplet(rec);

				rec.nodes[0] = x.id;
				rec.nodes[1] = z.id;
				rec.nodes[2] = y.id;
				rec.labels[0] = x.label;
				rec.labels[1] = z.label;
				rec.labels[2] = y.label;
				rec.true_class = 1;
				lets.push_back(rec);
				//cout << "Insert rec for z = " << z << endl;
				//print_triplet(rec);

				rec.nodes[0] = z.id;
				rec.nodes[1] = x.id;
				rec.nodes[2] = y.id;
				rec.labels[0] = z.label;
				rec.labels[1] = x.label;
				rec.labels[2] = y.label;
				rec.true_class = 0;
				lets.push_back(rec);
				//cout << "Insert rec for z = " << z << endl;
				//print_triplet(rec);
			}
		}
	}

	for(auto x: rnodes){
		for(auto y: rnodes){
			if(y.id == x.id) continue;
			for(auto z: lnodes){ //(x,y) and (y,x) will happen separately..
				pred_rec rec;
				rec.score = 1.0;
				rec.nodes[0] = x.id;
				rec.nodes[1] = y.id;
				rec.nodes[2] = z.id;
				rec.labels[0] = x.label;
				rec.labels[1] = y.label;
				rec.labels[2] = z.label;
				rec.true_class = 2;
				lets.push_back(rec);
				//cout << "Insert rec for z = " << z << endl;
				//print_triplet(rec);


				rec.nodes[0] = x.id;
				rec.nodes[1] = z.id;
				rec.nodes[2] = y.id;
				rec.labels[0] = x.label;
				rec.labels[1] = z.label;
				rec.labels[2] = y.label;
				rec.true_class = 1;
				lets.push_back(rec);
				//cout << "Insert rec for z = " << z << endl;
				//print_triplet(rec);

				rec.nodes[0] = z.id;
				rec.nodes[1] = x.id;
				rec.nodes[2] = y.id;
				rec.labels[0] = z.label;
				rec.labels[1] = x.label;
				rec.labels[2] = y.label;
				rec.true_class = 0;
				lets.push_back(rec);
				//cout << "Insert rec for z = " << z << endl;
				//print_triplet(rec);
			}
		}
	}
	nodes.insert(nodes.end(), lnodes.begin(), lnodes.end());
	nodes.insert(nodes.end(), rnodes.begin(), rnodes.end());
}

node_rec * get_node(string rec){

	//cout << "get node inp : " << rec << endl;

	node_rec * t;
	size_t pos = rec.find_first_of("_");
	if(pos == string::npos){
		t = new node_rec();
		t->left = NULL;
		t->right = NULL;
		t->id = atoi(rec.c_str());
		return t;
	}

	t = new node_rec();
	t->left = NULL;
	t->right = NULL;
	t->id = atoi(rec.substr(0, pos).c_str());

	size_t pos2 = rec.find_first_of(":");
	if(pos2 != string::npos){
		t->label = rec.substr(pos+1, (pos2-pos-1));
	}

	//cout << "See: id : " << t->id << " and label : " << t->label << endl;
	
	return t;

}

node_rec * CTree::parse_tree(string & trstr, int & idx){	 //assuming binary tree..
	node_rec * t1;
	node_rec * t;
	node_rec * t2 = NULL;
	int len = trstr.length();
	//cout << "SEEE : " << trstr.substr(idx, string::npos) << endl;
	if(trstr[idx] == '('){
		++idx;
		t1 = parse_tree(trstr, idx);

		if(trstr[idx] == ','){ //one more..
			++idx;
			t2 = parse_tree(trstr, idx);
		}


		//Now it has to be ')'
		++idx; //skip the matching ')'
		while((trstr[idx] != ')') && (trstr[idx] != ',') && (idx < len)) { ++idx; } //skip the unwanted for the ')'..

		if(t2 != NULL){
			t = new node_rec();
			t->left = t1;
			t->right = t2;
		}
		return t;
	}
	//leaf node.
	int idx2;
	for(idx2=idx; idx2 < trstr.length() ; ++idx2){
		if((trstr[idx2] == ')') || (trstr[idx2] == ',')) break; 
	}

	//cout << "Calling get node .. " << endl;
	//cout << "SEEE N: " << trstr.substr(idx, string::npos) << endl;
	t = get_node(trstr.substr(idx, (idx2 - idx)));
	/*
	t = new node_rec();
	t->left = NULL;
	t->right = NULL;
	t->id = atoi(trstr.substr(idx, (idx2 - idx)).c_str());
	*/
	//cout << "See : " << trstr.substr(idx, (idx2 - idx)) <<  " , " << idx2 << " , " << idx << endl;
	idx = idx2;
	//cout << "Here got " << t-> id << endl;
	
	return t;
}

string CTree::normalize_tree(node_rec * T){ //for exact tree maching invariant to left/right flips..

	if((T->right == NULL) && (T->left == NULL)){
		string str = to_string(T->id);
		//cout << "SEE: " <<  str << endl;
		return str;
	}

	string str1 = normalize_tree(T->left);
	string str2 = normalize_tree(T->right);
	if(str2.compare(str1) < 0){ //flip
		node_rec * tmp = T->left;
		T->left = T->right;
		T->right = tmp;
		//cout << "SEE: " <<  str2+"#"+str1 << endl;
		return str2+"#"+str1;
	}
	//cout << "SEE: " <<  str1+"#"+str2 << endl;
	return str1+"#"+str2;

}

void gen_triplet_from_file(string fname){


		//string test = "((((1,2),(3,4)),(5,6)),((8,9),10))";
		//string test = "(((((3,4),2),1),((5,10),6)),(8,9))";
		//string test = "(((((3,4),2),1),((5,10),6)),(8,9))";
		//string test = "\"(((1_2012212021:0.352885614,(2_2112212021:0.2878748513,(3_2112212021:0.2878748513,4_2112212021:0.2878748513):0):0.06501076267):0.3028150686,((5_0012212221:0.3476539115,10_0112212221:0.3476539115):0.1021496084,6_0012012221:0.4498035199):0.2058971627):0.2292344581,(8_2120010021:0.2839299162,9_2120010021:0.2839299162):0.6010052244);\"";
		//string test = "(((1_2012212021:0.352885614,(2_2112212021:0.2878748513,(3_2112212021:0.2878748513,4_2112212021:0.2878748513):0):0.06501076267):0.3028150686,((5_0012212221:0.3476539115,10_0112212221:0.3476539115):0.1021496084,6_0012012221:0.4498035199):0.2058971627):0.2292344581,(8_2120010021:0.2839299162,9_2120010021:0.2839299162):0.6010052244)";

		//string test = "((((1_2012212021:8,2_2112212021:8):38,(3_2112212021:4,4_2112212021:4):42):42,(5_0012212221:1,6_0012012221:1):87):46,((8_2120010021:22,9_2120010021:22):51,10_0112212221:74):62)"; 
		//string test = "(1,(2,3))";

	ifstream treef;
	treef.open(fname);
	string fline;
	CTree ct;

	cout << "dreamID,barcode,triplet_cell1,triplet_cell2,triplet_cell3,true_class,predicted_class,prob" << endl;
	//skip first line.
	getline(treef, fline);
	
	while(getline(treef, fline)){
		vector<string> line_arr = split(fline, '\t');
		string test = line_arr[3];
		test = test.substr(1,test.length()-3);
		//cout << "Test to process: \n" << test << endl;

		int idx = 0;
		int dreamid = atoi(line_arr[0].c_str());
		node_rec * t = ct.parse_tree (test, idx);
		ct.normalize_tree(t);

		//cout << "Input : " << test << endl;

		stringstream ss1;
		ct.print_tree_newick(t, ss1);

		cout << "Tree : " << ss1.str() << endl;

		vector<pred_rec> recs;
		vector<node_rec> nodes;

		ct.gen_triplets(t, recs, nodes);
		for(auto x: recs){
			cout << dreamid << "," << x.labels[0] << "_" << x.labels[1] << "_" << x.labels[2] << "," << x.nodes[0] << "," << x.nodes[1] << "," << x.nodes[2] << "," << (x.true_class + 1) <<  endl;
		}
	}

}

int main(int argc, char** argv){



	bool v1 = false;

	if(argc < 3){
		cout << "Usage: \nctree -c <triplet_file>\nctree -g <newick_tree_file>" << endl; 
		cout << "Usage: ctree -c <triplet_file> -v1" << endl; 
		exit(1);
	}
        vector<string> args(argv+1, argv + argc);

	bool gen_tripet = false;
	if(args[0] == "-g") {
		gen_tripet = true;
	}else if ((argc >= 4 ) && (args[2] == "-v1")) {
		v1 = true; //old version of the code..
	}
	
	

	if(gen_tripet){

		cout << "Triplet generation... " << endl;
		gen_triplet_from_file(args[1]);

		return 0;
	}


	ifstream tripf;
	tripf.open(args[1]); 

	ofstream fch;
	fch.open("trees_submission.txt");

	//Header
	fch << "dreamID\tnw\n";

	ofstream fout;
	fout.open("trees_output.txt");

	bool done = false;
	int oldDream = -1;
	int count = 0;

	if(v1){
		cout << "Tree reconstruction (v1 code) ... " << endl;
	}else{
		cout << "Tree reconstruction (latest separation code) ... " << endl;
	}

	while(!done){
		CTree ct;
		ct.dreamID = oldDream;
		if(v1) ct.best_pivot_maj = true;
		//cout << "Loading triplet prediction ..." << endl;
		done = ct.load_rf_triplet_predictions(tripf);
		//cout << "Tree construction ..." << endl;

		node_rec * T = ct.construct_tree();
		ct.normalize_tree(T);
		stringstream ss;
		ct.print_tree_newick(T, ss);
		fout << ">Dream id: " << ct.dreamID << endl;
		fout << "Output tree : \n" << ss.str() << endl; 

		stringstream ss2;
		ct.print_tree_newick(T, ss2, true);
		fch << ct.dreamID << "\t";
		fch << ss2.str() << "root;" << endl; 
		oldDream = ct.dreamID;
		++count;
		cout << "\rDone tree " << count << " ..." << flush;
	}
	cout << "\nOutput trees written to trees_output.txt" << endl;
	cout << "Output trees in challenge format written to trees_submission.txt" << endl;
	tripf.close();
	fch.close();
	fout.close();
}

