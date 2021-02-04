import sys
import csv
import random
import operator
from collections import Counter


TEST_SET_SIZE = 100
VERBOSE = False


def read_data(train_file, predict_col_name):
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
            name__votes[name] = row[4:]
            name__class[name] = class_val
    if VERBOSE:
        print("read", len(name__votes), "rows")
    return name__votes, name__class


def predict_votes(train_vote_lists, test_vote_lists, class_vals, k):
    correct_count = 0
    for test_name, test_votes in test_vote_lists.items():
        if VERBOSE:
            print("predicting vote for ", test_name)
        predict_class = predict(train_vote_lists, test_votes, class_vals, k)
        if VERBOSE:
            print("predicted value:", predict_class, "actual value:", class_vals[test_name])
        if predict_class == class_vals[test_name]:
            correct_count += 1
    accuracy = correct_count / len(test_vote_lists)
    if VERBOSE:
        print("correct {}/{} = {:.2f}".format(correct_count, len(test_vote_lists), accuracy))
    return accuracy


def predict(train_dict, test_row, class_dict, k):
    nearest_neighbors = knn(train_dict, test_row, k)
    vote_results = []
    for k,v in class_dict.items():
       for i in nearest_neighbors:
           if i == k:
               vote_results.append(v)
               
   # for i in range(len(vote_results)):
   #     if vote_results[i] == 'Yea':
   #         vote_results[i] = 'Yes'
   #     if vote_results[i] == 'Nay':
   #         vote_results[i] = 'No'
   #     if vote_results[i] == 'Present':
   #         vote_results[i] = 'Not Voting'
    
    vote_amt = Counter(vote_results)
    result = list(vote_amt.keys())[0]
    return result


def knn(train_dict, test_row, k):

    neighbors = []
    temp_dictionary = {}
    temp_row = list(train_dict.items())
    for i in temp_row:
       temp_name = i[0]
       temp_distance = distance(test_row, i[1])
       temp_dictionary.update({temp_name : temp_distance})

    sorted_reps = sorted(temp_dictionary.items(), key =operator.itemgetter(1))
    for i in range(k):
        neighbors.append(sorted_reps[i][0])
     
    return neighbors

        
def distance(row1, row2):
    distance_num = 0
    for i in range(len(row1)):
        if ((row1[i] == row2[i])):
            next
        elif ((row1[i] == 'Not Voting' or row1[i] == 'Present') and (row2[i] == 'Not Voting' or row2[i] == 'Present')):
            next
        elif ((row1[i] == 'Yea' or row1[i] == 'Yes' or row1[i] == 'Aye') and (row2[i] == 'Yea' or row2[i] == 'Yes' or row2[i] == 'aye')):
            next   
        elif ((row1[i] == 'No' or row1[i] == 'Nay') and (row2[i] == 'No' or row2[i] == 'Nay')):
            next
        elif ((row1[i] == '' and row2[i] != '') or (row1[i] != '' and row2[i] == '')):
            distance_num += 10
            next
        else:
            distance_num+=10
            
    return distance_num


###################################

if __name__ == '__main__':

    
    if len(sys.argv) <= 3:
        sys.exit("USAGE: " + sys.argv[0] + " path/to/congress_data.csv predict_col_name k")
    train_file_path = sys.argv[1]
    predict_col = sys.argv[2]
    k = int(sys.argv[3])

    # read in the data file
    votes, class_vals = read_data(train_file_path, predict_col)

    #split the data into training and test sets, putting into dictionaries keyed by name
    all_names = list(votes.keys())
    random.shuffle(all_names)
    train_names = all_names[:-TEST_SET_SIZE]
    test_names = all_names[-TEST_SET_SIZE:]
    train_vote_lists = {name: votes[name] for name in train_names}
    test_vote_lists = {name: votes[name] for name in test_names}

    # run the classifier and print the accuracy
    acc = predict_votes(train_vote_lists, test_vote_lists, class_vals, k)
    print(acc)
