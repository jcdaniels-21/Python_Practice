#!/usr/bin/python
import sys
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import operator

def main():
    df = read_data()
    type_distirbution(df)
    top_5(df)
    stat_distribution_plot(df)
    stat_by_generation(df)
    legendaries(df)
    comparisons(df)

def main2():
    df = read_data()
    x, y = data_prep(df)
    imputed_pokemon_data = impute_data(x)
    x_train, x_test, y_train, y_test = train_test_split(imputed_pokemon_data, y, test_size=0.25)
    train_pos_count, train_neg_count = legendary_distribution(y_train)
    test_pos_count, test_neg_count = legendary_distribution(y_test)
    print("\ntraining class distrib: {:.2f}, {:.2f}".format(train_pos_count/len(y_train), 
                                                          train_neg_count/len(y_train)))
    print("\ntest class distrib: {:.2f}, {:.2f}".format(test_pos_count/len(y_test),
                                                      test_neg_count/len(y_test)))
    
    tree_clf = learn_tree(x_train, y_train)
    acc_train = test_model(tree_clf, x_train, y_train)
    acc_test = test_model(tree_clf, x_test, y_test)
    print("\ndecision tree training acc: {:.4f}, test acc: {:.4f}".format(acc_train, acc_test))
    
    print("\n Top 3 Features")
    for feat, score in top_features(tree_clf, x.columns, 3):
        print("{} ({:.4f})".format(feat, score))
    
    print("\nknn testing at various values of n")
    k_vals = [1, 3, 5, 7, 9, 19, 39, 79, 159, 319]
    for k in k_vals:
        knn_clf = learn_knn(x_train, y_train, k)
        acc_test = test_model(knn_clf, x_test, y_test)
        print("k-nn {} test acc: {:.4f}".format(k, acc_test))
        
    acc_test = crossval_tree(imputed_pokemon_data, y, 5)
    print("\ndecision tree {}-fold cross-validation acc: {:.4f}".format(5, acc_test))    
  

def main3(mon_name):
    pokemon, class_vals = read_data2('pokemon.csv', 'pokedex_number')
    your_poke_values = [pokemon[key] for key in pokemon if key == mon_name]
    poke_neighbors = knn(pokemon, your_poke_values[0], 6)
    poke_neighbors = poke_neighbors[1:]
    print("\n Pokemon closest to {}".format(mon_name))
    print(poke_neighbors)
    
 
def main4():
    main()
    main2()
    main3('Squirtle')     
    
    

def read_data():
    poke_df = pd.read_csv('pokemon.csv')
    df = pd.DataFrame()
    df['Pokedex Number'] = poke_df['pokedex_number']
    df['Name'] = poke_df['name']
    df['Type 1'] = poke_df['type1']
    df['Type 2'] = poke_df['type2']
    df['Base Total'] = poke_df['base_total']
    df['HP'] = poke_df['hp']
    df['Attack'] = poke_df['attack']
    df['Defense'] = poke_df['defense']
    df['Sp. Atk'] = poke_df['sp_attack']
    df['Sp. Def'] = poke_df['sp_defense']
    df['Speed'] = poke_df['speed']
    df['Height (m)'] = poke_df['base_total']
    df['Weight (kg)'] = poke_df['weight_kg']
    df['Generation'] = poke_df['generation']
    df['Legendary'] = poke_df['is_legendary']
    return df

def type_distirbution(df):
    print("\nPrimary type distribution:")
    type1_count = pd.DataFrame(df['Type 1'].value_counts())
    type1_count =type1_count.reset_index()
    type1_count.columns = ['Type', 'Count']
    for index, row in type1_count.iterrows():
        print('{}: {}'.format(row['Type'], row['Count']))
  
    print("\nSecondary type distribution:")
    type2_count = pd.DataFrame(df['Type 2'].value_counts())
    type2_count =type2_count.reset_index()
    type2_count.columns = ['Type', 'Count']
    for index, row in type2_count.iterrows():
        print('{}: {}'.format(row['Type'], row['Count']))
        
def top_5(df):
    print("\n: Top 5 Pokemon by Base Health")
    temp = df
    temp = temp.sort_values(['HP'], ascending =False)
    hp_pokemon = temp['Name'][:5]
    hp_stats = temp['HP'][:5]
    hp = pd.DataFrame()
    hp['Pokemon'] = hp_pokemon
    hp['Base Health'] = hp_stats
    print(hp)
    
    print("\n: Top 5 Pokemon by Base Attack")
    temp2 = df
    temp2 = temp2.sort_values(['Attack'], ascending =False)
    atk_pokemon = temp2['Name'][:5]
    atk_stats = temp2['Attack'][:5]
    atk = pd.DataFrame()
    atk['Pokemon'] = atk_pokemon
    atk['Base Attack'] = atk_stats
    print(atk)
    
    print("\n: Top 5 Pokemon by Base Defense")
    temp3 = df
    temp3 = temp3.sort_values(['Defense'], ascending =False)
    def_pokemon = temp3['Name'][:5]
    def_stats = temp3['Defense'][:5]
    defense = pd.DataFrame()
    defense['Pokemon'] = def_pokemon
    defense['Base Defense'] = def_stats
    print(defense)
    
    print("\n: Top 5 Pokemon by Base Sp. Attack")
    temp4 = df
    temp4 = temp.sort_values(['Sp. Atk'], ascending =False)
    sp_atk_pokemon = temp4['Name'][:5]
    sp_atk_stats = temp4['Sp. Atk'][:5]
    sp_atk = pd.DataFrame()
    sp_atk['Pokemon'] = sp_atk_pokemon
    sp_atk['Base Sp. Atk'] = sp_atk_stats
    print(sp_atk)
    
    print("\n: Top 5 Pokemon by Base Sp. Defense")
    temp5 = df
    temp5 = temp5.sort_values(['Sp. Def'], ascending =False)
    spdef_pokemon = temp5['Name'][:5]
    spdef_stats = temp5['Sp. Def'][:5]
    spdef = pd.DataFrame()
    spdef['Pokemon'] = spdef_pokemon
    spdef['Base Sp. Def'] = spdef_stats
    print(spdef)
    
    print("\n: Top 5 Pokemon by Base Speed")
    temp6 = df
    temp6 = temp6.sort_values(['Speed'], ascending =False)
    speed_pokemon = temp6['Name'][:5]
    speed_stats = temp6['Speed'][:5]
    speed = pd.DataFrame()
    speed['Pokemon'] = speed_pokemon
    speed['Base Speed'] = speed_stats
    print(speed)
    
def stat_distribution_plot(df):
    df = df.drop(['Pokedex Number', 'Name', 'Type 1', 'Type 2', 'Base Total', 'Height (m)', 'Weight (kg)', 'Generation', 'Legendary'], 1)
    stat_plot = sns.boxplot(data = df)
    stat_plot.set_title("Stat Distribution Across All Pokemon")
    stat_plot.set_ylabel('Base Value')
    plt.show()
    
def stat_by_generation(df):
    hp_list = df['HP']
    hp_dict = {}
    atk_list = df['Attack']
    atk_dict = {}
    def_list = df['Defense']
    def_dict = {}
    sp_atk_list = df['Sp. Atk']
    sp_atk_dict = {}
    sp_def_list = df['Sp. Def']
    sp_def_dict = {}
    speed_list = df['Speed']
    speed_dict = {}
    generations = df['Generation']
   
    
    for i in df['Generation']:
        if i in hp_dict:
            continue
        else:
            hp_dict.update({i: []})
            atk_dict.update({i: []})
            def_dict.update({i: []})
            sp_atk_dict.update({i: []})
            sp_def_dict.update({i: []})
            speed_dict.update({i: []})
    
    for i in range(len(generations)):
        hp_dict[generations[i]].append(hp_list[i])
        atk_dict[generations[i]].append(atk_list[i])
        def_dict[generations[i]].append(def_list[i])
        sp_atk_dict[generations[i]].append(sp_atk_list[i])
        sp_def_dict[generations[i]].append(sp_def_list[i])
        speed_dict[generations[i]].append(speed_list[i])
        
        
    avgHP = {}
    for k,v in hp_dict.items():
        avgHP[k] = sum(v)/ float(len(v))
    
    hp_graph_list = avgHP.items()
    hp_x, hp_y = zip(*hp_graph_list)
    plt.plot(hp_x, hp_y, c='Red', marker='o' )
    plt.title('Average Base HP by Generation', loc='center')
    plt.xlabel("Generation")
    plt.ylabel("Base Value")
    plt.show()

    avgAtk = {}
    for k,v in atk_dict.items():
        avgAtk[k] = sum(v)/ float(len(v))
    
    atk_graph_list = avgAtk.items()
    atk_x, atk_y = zip(*atk_graph_list)
    plt.plot(atk_x, atk_y, c='Orange', marker='o' )
    plt.title('Average Base Attack by Generation', loc='center')
    plt.xlabel("Generation")
    plt.ylabel("Base Value")
    plt.show()
    
    avgDef = {}
    for k,v in def_dict.items():
        avgDef[k] = sum(v)/ float(len(v))
    
    def_graph_list = avgDef.items()
    def_x, def_y = zip(*def_graph_list)
    plt.plot(def_x, def_y, c='Yellow', marker='o' )
    plt.title('Average Base Defense by Generation', loc='center')
    plt.xlabel("Generation")
    plt.ylabel("Base Value")
    plt.show()
    
    avgSpAtk = {}
    for k,v in sp_atk_dict.items():
         avgSpAtk[k] = sum(v)/ float(len(v))
    
    spatk_graph_list = avgSpAtk.items()
    spa_x, spa_y = zip(*spatk_graph_list)
    plt.plot(spa_x, spa_y, c='Blue', marker='o' )
    plt.title('Average Base Special Attack by Generation', loc='center')
    plt.xlabel("Generation")
    plt.ylabel("Base Value")
    plt.show()
    
    avgSpDef = {}
    for k,v in sp_def_dict.items():
         avgSpDef[k] = sum(v)/ float(len(v))
    
    spdef_graph_list = avgSpDef.items()
    spdef_x, spdef_y = zip(*spdef_graph_list)
    plt.plot(spdef_x, spdef_y, c='Green', marker='o' )
    plt.title('Average Base Special Defense by Generation', loc='center')
    plt.xlabel("Generation")
    plt.ylabel("Base Value")
    plt.show()
    
    avgSpeed = {}
    for k,v in speed_dict.items():
         avgSpeed[k] = sum(v)/ float(len(v))
    
    speed_graph_list = avgSpeed.items()
    spd_x, spd_y = zip(*speed_graph_list)
    plt.plot(spd_x, spd_y, c='Pink', marker='o' )
    plt.title('Average Base Speed by Generation', loc='center')
    plt.xlabel("Generation")
    plt.ylabel("Base Value")
    plt.show()


def legendaries(df):
    all_mons = df['Legendary']
    generations = df['Generation']
    names = df['Name']
    legendaries = []
    for i in range(len(all_mons)):
        if all_mons[i] == 1:
            data = (names[i], generations[i])
            legendaries.append(data)
    
    temp_dict = {}    
    for mons in legendaries:   
        if mons[1] in temp_dict:
            temp_dict[mons[1]].append(mons[0])
        else:
            temp_dict.update({mons[1]: []})
            temp_dict[mons[1]].append(mons[0])
    
    
    for k, v in temp_dict.items():
        print("Generation: {}, Legendaries: {}".format(k, v))
        
    
    
    legend_count = {}
    for k, v in temp_dict.items():
        legend_count[k] = len(v) 
    
    legend_count_list = legend_count.items()
    x, y = zip(*legend_count_list)
    plt.bar(x, y, color=['#CC0030', '#F24C0C', '#0D4448', '#D7A801', '#7b9095', '#367588', '#0B6623'], edgecolor = 'black')
    plt.title('Count of Legendary Pokemon by Generation')
    plt.xlabel("Gneration")
    plt.ylabel("Count")
    plt.show()
    
def comparisons(df):
    non_legendary = df[df.Legendary == 0]
    legendary  = df[df.Legendary == 1]
    stats = ['Base Total', 'HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']
    legendary_stats = [legendary['Base Total'].mean(), legendary['HP'].mean(), legendary['Attack'].mean(), legendary['Defense'].mean(), legendary['Sp. Atk'].mean(), legendary['Sp. Def'].mean(), legendary['Speed'].mean()]
    non_legendary_stats = [non_legendary['Base Total'].mean(), non_legendary['HP'].mean(), non_legendary['Attack'].mean(), non_legendary['Defense'].mean(), non_legendary['Sp. Atk'].mean(), non_legendary['Sp. Def'].mean(), non_legendary['Speed'].mean()]
    
    print("\nAverage stats for non legndary pokemon")
    for i in range(len(non_legendary_stats)):
        print("{} : {:.2f}".format(stats[i], non_legendary_stats[i]))
    print("\nAverage stats for legendary pokemon")
    for i in range(len(legendary_stats)):
        print("{} : {:.2f}".format(stats[i], legendary_stats[i]))
    
####### PART 2 #############
def data_prep(df):
    x = df
    y = x['Legendary']
    del x['Legendary']
    del x['Pokedex Number']
    del x['Name']
    del x['Type 1']
    del x['Type 2']
    return x, y


def impute_data(x):
    fill_empty = SimpleImputer(missing_values = np.nan, strategy='mean')
    imputed_pokemon = x  
    for col in imputed_pokemon:
        imputed_col = fill_empty.fit_transform(imputed_pokemon)
        imputed_pokemon[col] = imputed_col
    imputed_pokemon = imputed_pokemon.to_numpy()
    return imputed_pokemon

def legendary_distribution(class_labels):
    legendary = 0
    regular = 0
    for label in class_labels:
        if label == 1:
            legendary+=1
        else:
            regular+=1
    return legendary, regular

def learn_tree(x, y):
    pokemon_tree = tree.DecisionTreeClassifier()
    pokemon_tree = pokemon_tree.fit(x, y)
    return pokemon_tree

def test_model(clf, x, y):
    acc = clf.score(x, y)
    return acc


def sortSecond(val):
    return val[1]

def top_features(clf, col_names, num):
    importances = clf.feature_importances_
    merged_list = [(col_names[i], importances[i]) for i in range(0, len(importances))]
    merged_list.sort(key = sortSecond, reverse = True)
    final_list = []
    for i in range(num):
        final_list.append(merged_list[i])
    return final_list

def learn_knn(x, y, k):
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(x, y)
    return clf

def crossval_tree(x, y, folds):
    classifier = tree.DecisionTreeClassifier()
    acc = cross_val_score(classifier, x, y, cv = folds)
    acc = np.mean(acc)
    return acc

###########Part 3################################
def read_data2(file_name, predict_col_name):
    name__stat = {}
    name__class = {}
    with open(file_name, encoding = 'utf-8') as infile:
        reader = csv.reader(infile)
        col_names = next(reader)
        class_col_idx = col_names.index(predict_col_name)
        for row in reader:
            name = row[1]
            class_val = row[class_col_idx]
            row.pop(class_col_idx)
            name__stat[name] = row[1:10]
            name__class[name] = class_val
    
    return name__stat, name__class

def distance(row1, row2):
    distance_num = 0
    for i in range(len(row1)):
        if i > 1:
            row1[i] = int(row1[i])
            row2[i] = int(row2[i])
    
    for i in range(len(row1)):
        if i == 0:
            if ((row1[i] == row2[i])):
                continue
            else:
                distance_num += 200
        elif i == 1:
            if ((row1[i] == row2[i])):
                continue
            elif ((row1[i] == '' and row2[i] != '') or (row1[i] != '' and row2[i] == '')):
                distance_num += 100
            else:
                distance_num += 200
        else:
            distance_num += abs(row1[i]-row2[i])
        
    return distance_num
            
def knn(train_dict, test_row, k):
    neighbors = []
    temp_dictionary = {}
    temp_row = list(train_dict.items())
    for i in temp_row:
       temp_name = i[0]
       temp_distance = distance(test_row, i[1])
       temp_dictionary.update({temp_name : temp_distance})

    sorted_pokemon = sorted(temp_dictionary.items(), key =operator.itemgetter(1))
    for i in range(k):
        neighbors.append(sorted_pokemon[i][0])
     
    return neighbors


    



#########################

if __name__ == '__main__':
    main4()