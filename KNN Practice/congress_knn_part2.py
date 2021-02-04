# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:33:45 2019

@author: Jarrod Daniels
"""
import sys
import csv
import random
import operator
from collections import Counter
import matplotlib.pyplot as plt
from congress_knn import read_data
from congress_knn import predict_votes
from congress_knn import predict
from congress_knn import knn
from congress_knn import distance

TEST_SET_SIZE = 100
VERBOSE = False


def main():
    k_acc = accuracy(13)
    visualize_acc(k_acc)
    



def accuracy(k):
    k_acc = {}
    i = 45
    while i > 13:
        votes, class_vals = read_data('congress_data.csv', 'Vote199')
        all_names = list(votes.keys())
        random.shuffle(all_names)
        train_names = all_names[:-TEST_SET_SIZE]
        test_names = all_names[-TEST_SET_SIZE:]
        train_vote_lists = {name: votes[name] for name in train_names}
        test_vote_lists = {name: votes[name] for name in test_names}
        acc = predict_votes(train_vote_lists, test_vote_lists, class_vals, k)
        k_acc.update({k : acc})
        k+=10
        i-=1
    #print(k_acc)
    return k_acc;


def visualize_acc(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    
    plt.figure(figsize =(11,11))
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.plot(keys, values)
    plt.show()
    
def knn_JimMcgovern():
    votes, class_vals = read_data('congress_data.csv', 'Vote1')
    jim_votes = [votes[key] for key in votes if key == 'James “Jim” McGovern']
    jim_neighbors = knn(votes, jim_votes[0], 11)
    jim_neighbors = jim_neighbors[1:]
    print(jim_neighbors)
    #return(jim_neighbors)

def read_data2(train_file, predict_col_name):
    name__votes = {}
    name__class = {}
    with open(train_file, encoding = 'utf-8') as infile:
        reader = csv.reader(infile)
        col_names = next(reader)  # header row
        class_col_idx = col_names.index(predict_col_name)
        for row in reader:  # Name, State, District, Party, Vote1, Vote2, ...
            name = row[0]
            class_val = row[class_col_idx]
            row.pop(class_col_idx)
            name__votes[name] = row[3:]
            name__class[name] = class_val
    if VERBOSE:
        print("read", len(name__votes), "rows")
    return name__votes, name__class


def irregular_candidates():
    votes, class_vals = read_data2('congress_data.csv', 'Vote1')
    all_voter_list = [(k, v) for k, v in votes.items()]
    republicans = []
    democrats = []
    for i in all_voter_list:
        if i[1][0] == 'Republican':
            republicans.append(i)
        elif i[1][0] == 'Democrat':
            democrats.append(i)
    for r in republicans:
        del r[1][0]
    for d in democrats:
        del d[1][0]
    
    irregular_republicans = []
    republican_distances = {}
    for r in republicans:
        temp_name = r[0]
        temp_vote_list = r[1]
        distance_away = 0
        for p in republicans:
            comparing_vote_list = p[1]
            temp_distance = distance(temp_vote_list, comparing_vote_list)
            distance_away+= temp_distance
        distance_away = distance_away/(len(republicans)-1)
        republican_distances.update({temp_name : distance_away})
    sorted_republicans = sorted(republican_distances.items(), key =operator.itemgetter(1), reverse = True)
    for i in range(3):
        irregular_republicans.append(sorted_republicans[i][0])
    
    irregular_democrats = []
    democrat_distances = {}
    for d in democrats:
        temp_name = d[0]
        temp_vote_list = d[1]
        distance_away = 0
        for p in democrats:
            comparing_vote_list = p[1]
            temp_distance = distance(temp_vote_list, comparing_vote_list)
            distance_away+= temp_distance
        distance_away = distance_away/(len(democrats)-1)
        democrat_distances.update({temp_name : distance_away})
    sorted_democrats = sorted(democrat_distances.items(), key =operator.itemgetter(1), reverse = True)
    for i in range(3):
        irregular_democrats.append(sorted_democrats[i][0])
    
    
    print("Three most irregular republicans: {}".format(irregular_republicans))
    print("Three most irregular democrats: {}".format(irregular_democrats))
    
    
    return(irregular_republicans,irregular_democrats)
    
    


###################################


if __name__ == '__main__':
    data_file_name = sys.argv[1]
    main(data_file_name)