#!/usr/bin/env python3
"""Score subchallenge 1"""
import argparse
import json

import pandas as pd
import math
import score


def main(submissionfile, goldstandard, results, path_to_treecmp):
    """Get scores and write results to json

    Args:
        submissionfile: Participant submission file path
        goldstandard: Goldstandard file path
        results: File to write results to
        path_to_treecmp: Path to TreeCmp
    """
    score_dict = {}
    prediction_file_status = "SCORED"
    submissiondf = pd.read_csv(submissionfile, sep="\t")
    goldstandarddf = pd.read_csv(goldstandard, sep="\t")
    # Match dreamID
    mergeddf = submissiondf.merge(goldstandarddf, on="dreamID")

    print(mergeddf.columns)

    rf_scores = []
    triple_scores = []
    scores_per_tree = []
    for _, row in mergeddf.iterrows():
        with open("truth.nwk", 'w') as truth:
            truth.write(row['ground'])
        with open("sub.nwk", 'w') as sub:
            sub.write(row['nw'])
        rooted_submission_path = score.reroot_submission("sub.nwk")
        # scores = score.get_scores("truth.nwk", submissionfile,
        #                           "treecmp_results.out", path_to_treecmp)
        
        scores = score.get_scores("truth.nwk", rooted_submission_path,
                                  "treecmp_results.out", path_to_treecmp)
        
        tree_scores = [str(x) for x in (row['dreamID'],
                                        0.0 if math.isnan(scores.T[0].loc['R-F_Cluster_toYuleAvg']) else scores.T[0].loc['R-F_Cluster_toYuleAvg'],
                                        0.0 if math.isnan(scores.T[0].loc['Triples_toYuleAvg']) else scores.T[0].loc['Triples_toYuleAvg'])]
        
        print('*********\n', tree_scores)
        
        scores_per_tree.append("\t".join(tree_scores))
        rf_scores.append(0 if math.isnan(scores.T[0].loc['R-F_Cluster_toYuleAvg']) else scores.T[0].loc['R-F_Cluster_toYuleAvg'])
        triple_scores.append(min(1, scores.T[0].loc['Triples_toYuleAvg']))

    score_dict['RF_average'] = sum(rf_scores) / len(rf_scores)
    score_dict['Triples_average'] = sum(triple_scores) / len(triple_scores)
    score_dict['prediction_file_status'] = prediction_file_status
    
    with open(results+'.colony','w') as rc:
        rc.write("id\tRF\ttriplet\n")
        rc.write("\n".join(scores_per_tree))
    
    with open(results, 'w') as o:
        o.write(json.dumps(score_dict))

    rc.close()
    o.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--submissionfile", required=True,
                        help="Submission File")
    parser.add_argument("-r", "--results", required=True,
                        help="Scoring results")
    parser.add_argument("-g", "--goldstandard", required=True,
                        help="Goldstandard for scoring")
    parser.add_argument("-p", "--treecmp", required=True,
                        help="Path to treecmp")
    args = parser.parse_args()
    main(args.submissionfile, args.goldstandard, args.results, args.treecmp)
