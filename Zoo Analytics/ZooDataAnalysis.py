# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:01:31 2021

@author: Jarrod Daniels
"""
import pandas as pd
import seaborn as sns


def main():
   df = read_data()
   total_entries(df)
   unqiueSpecies(df)
   maxMinLifeExpect(df)
   avgLifeExpect(df) 
   whiskerboxPlot(df)
   genderBreakdown(df)
   MaxgenderDesparity(df)

   
def main2():
    df = read_data()
    genderMLEDesparity(df)
    

 
def read_data():
    df = pd.read_csv('E:\Documents\GitHub\Python_Practice\AZA_MLE_Jul2018.csv', encoding= 'unicode_escape')
    df = df.drop(['Male Data Deficient', 'Female Data Deficient'], axis=1)
    return df

def total_entries(df):
    print("There are {} entries in the data set across the variables:".format(len(df)))
    for col in df.columns:
        print(col)
    for i in range(2):
        print("\n")
    
def unqiueSpecies(df):
    n = df.TaxonClass.value_counts()
    print("The taxonomic break down of the species in the data set is as follows:")
    print(n)
    print("\n")
    print("distirubtion wise the breakdown is as follows:")
    print(df.TaxonClass.value_counts(normalize = True))
    print("\n")
    
def maxMinLifeExpect(df):
    maximum = df.loc[df['Overall MLE'] == df['Overall MLE'].max()]
    maxVal = maximum[['Species Common Name', 'TaxonClass', 'Overall MLE']]
    print("Species with the highest overall MLE:")
    for col in maxVal.columns:
        print(col + ": {}".format(maxVal[col].to_string(index=False)))
    print("\n")
    
    minimum = df.loc[df['Overall MLE'] == df['Overall MLE'].min()]
    minVal = minimum[['Species Common Name', 'TaxonClass', 'Overall MLE']]
    print("Species with the lowest overall MLE:")
    for col in minVal.columns:
        print(col + ": {}".format(minVal[col].to_string(index=False)))
    print("\n")
    

def avgLifeExpect(df):
    print("Average MLE by Taxon Class: \n")
    df = df.groupby(['TaxonClass'])['Overall MLE'].mean().reset_index(name ='Avergae MLE')
    df = df.sort_values(by=['Avergae MLE'], ascending=False)
    df['Avergae MLE'] = df['Avergae MLE'].round(1)
    df = df.dropna()
    print(df.to_string(index=False))
    
def whiskerboxPlot(df):
    box_df = df[['TaxonClass', 'Overall MLE']]
    box_df = box_df.dropna()
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x="TaxonClass", y="Overall MLE", data=box_df)

def genderBreakdown(df):
    totalFemaleObservations = df['Female Sample Size '].sum()
    totalMaleObservations = df['Male Sample Size'].sum()
    print("Total population size across all samples in the data set {}".format(totalFemaleObservations+totalMaleObservations))
    print("Female population size across data set: {}. Proportion of total ~ {}%".format(totalFemaleObservations, round((totalFemaleObservations/(totalFemaleObservations+totalMaleObservations))*100)))
    print("Male population size across data set: {}. Proportion of total ~ {}%".format(totalMaleObservations, round((totalMaleObservations/(totalFemaleObservations+totalMaleObservations))*100)))


def genderDisbySpecies(df):
    dfFem = df.groupby(['TaxonClass'])['Female Sample Size '].sum().reset_index(name ='Female Sample Size')
    dfMal = df.groupby(['TaxonClass'])['Male Sample Size'].sum().reset_index(name ='Male Sample Size')
    cols_to_use = dfMal.columns.difference(dfFem.columns)
    combinedDF = pd.merge(dfFem, dfMal[cols_to_use],  left_index=True, right_index=True, how='outer')
    combinedDF['Percent Female'] = combinedDF['Female Sample Size']/(combinedDF['Female Sample Size'] + combinedDF['Male Sample Size'])*100
    combinedDF['Percent Male'] = combinedDF['Male Sample Size']/(combinedDF['Female Sample Size'] + combinedDF['Male Sample Size'])*100
    print(combinedDF)
    
def genderMLEDesparity(df):
    df['difference'] = abs((df['Female MLE'] - df['Male MLE']))
    df = df.sort_values(by=['difference'], ascending=False)
    df = df.dropna()
    print("Top 5 animals with greatest differing MLE (by Gender):")
    print(df[['Species Common Name', 'TaxonClass', 'difference']].head())
    print("\n")
    print("Top 5 animals with least differing MLE (by Gender):")
    print(df[['Species Common Name', 'TaxonClass', 'difference']][-5:])
    print("\n")
    avg_df = df.groupby(['TaxonClass'])['difference'].mean().reset_index(name ='Avg Difference in MLE')
    print(avg_df)
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x="TaxonClass", y="difference", data=df)
    ax.set_title('Difference in MLE Based on Gender (Grouped by Taxon Group)')
    ax.set_ylabel("Difference in MLE (years)")
    