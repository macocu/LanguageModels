#!/usr/bin/env python
# -*- coding: utf8 -*-

'''
Script that reads in output files and outputs some statistics:
average, highest score, lowest, std dev, etc
'''

import os
import argparse
import numpy as np
import scipy.stats as st  # For confidence interval


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, type=str,
                        help="Main evaluation folder")
    parser.add_argument("-o", "--output_file", default='', type=str,
                        help="Output file, if not added we just add 'avg_scores.txt' to args.f")
    parser.add_argument("-p", "--prefix", default='',
                        help="Prefix that all files should start with")
    args = parser.parse_args()
    return args


def get_full_files_in_folder(folder):
    '''Gets all files in a folder'''
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))]


def get_filename(fil):
    '''Get only the filename from full path filename'''
    return fil.split('/')[-1]


def average_list(in_list):
    '''From a list of numbers, return the average'''
    return float(sum(in_list)) / float(len(in_list))


class AverageScores:
    '''Class to keep track of the average score of experiments'''
    # Output name if output_file is empty
    output_name = "avg_scores.txt"

    def __init__(self, folder, output_file, prefix):
        self.folder = folder
        self.prefix = prefix
        self.output_file = output_file
        if not self.output_file:
            self.output_file = self.folder + self.prefix + '_' + self.output_name \
                               if self.prefix else self.folder + self.output_name
        # Set the IDs we check for calculating scores
        #self.line_ids = ['f1_score', 'precision', 'recall']
        self.line_ids = ["f1_score"]
        self.file_types = ["dev", "test"]
        # Save those results in a dictionary
        self.score_dict = {}
        for file_type in self.file_types:
            for ident in self.line_ids:
                self.score_dict[ident + " for " + file_type] = []
        # Get the evaluation files we will be using, based on the input folder
        self.eval_files = get_full_files_in_folder(self.folder)
        # Filter based on prefix
        if self.prefix:
            self.eval_files = [fil for fil in self.eval_files
                               if get_filename(fil).startswith(self.prefix)]
        assert self.eval_files, "No evaluation files found in folder {0}".format(folder)

    def get_scores(self, line_id):
        '''Return a list of scores based on the files and indicator of the line to find'''
        scores = []
        for eval_file in self.eval_files:
            for line in open(eval_file, 'r'):
                if line.strip().startswith(line_id):
                    scores.append(float(line.split()[-1]))
        return scores

    def calc_scores(self):
        '''Read in Counter or other output files and average all the scores'''
        # Get the F-scores and save all output information
        for line_id in self.score_dict:
            scores = self.get_scores(line_id)
            scores = [score * 100 for score in scores]
            # Calculate the statistics, on separate lines for clarity
            highest = max(scores)
            lowest = min(scores)
            average = average_list(scores)
            num_runs = len(scores)
            std_dev = round(np.std(scores), 2)
            conf_interval = [round(num, 4) for num in st.t.interval(0.95,
                             len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))]
            string_scores = ", ".join([str(x) for x in scores])
            # Save to list for specific line_id
            self.score_dict[line_id] = [num_runs, string_scores, highest,
                                        lowest, average, std_dev, conf_interval]

    def print_scores(self):
        '''Print the calculated scores to the output file.
           Always call calc_scores before this function'''
        with open(self.output_file, 'w') as out_f:
            for l_id in self.score_dict:
                num_runs, string_sc, highest, lowest, avg, std_dev, conf_int = self.score_dict[l_id]
                out_f.write("For score-type: {}\n".format(l_id))
                out_f.write("Runs: {}\n".format(num_runs))
                out_f.write("All scores: {}\n".format(string_sc))
                out_f.write("Highest score: {}\n".format(highest))
                out_f.write("Lowest score : {}\n".format(lowest))
                out_f.write("Average score: {}\n".format(avg))
                out_f.write("Average rounded score: {}\n".format(round(avg,1)))
                out_f.write("Standard deviation: {}\n".format(std_dev))
                out_f.write("Confidence interval: {}\n\n".format(conf_int))
        out_f.close()

if __name__ == "__main__":
    args = create_arg_parser()
    # Do the averaging and print to file
    avg_scores = AverageScores(args.folder, args.output_file, args.prefix)
    avg_scores.calc_scores()
    avg_scores.print_scores()
