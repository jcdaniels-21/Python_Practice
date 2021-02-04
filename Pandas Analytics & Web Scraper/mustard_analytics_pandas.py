#!/usr/bin/python
import sys
import pandas as pd
import calendar
  



#Main to execute all functions in the data
def main(file_name):
   df = read_data(file_name)
   total_cost(df)
   least_common_locs(df)
   most_common_locs(df)
   state_totals(df)
   unique_dates(df)
   month_avg(df)



#Read data
def read_data(file_name):
    df = pd.read_csv('mustard_data.csv')  # fix this!
    df['date'] = pd.to_datetime(df['date'])
    df['mileage'] = df['mileage'].fillna(0)
    df['mileage'] = df['mileage'].astype(int)
    df['cost'] = df['cost'].replace('[\$,]', '', regex=True).astype(float)
    return df


#Total Money Spent of Gas
def total_cost(df):
    print("\nExercise 1:")
    amt_spent = df.gallons * df.cost
    total = amt_spent.sum()
    total = round(total, 2)
    print('${}'.format(total))
   


#Refueling Locations Visited Once
def least_common_locs(df):
    print("\nExercise 2:")
    location_count = df.location.value_counts()
    unique_locations = location_count[location_count <= 1]
    print(unique_locations.count())


#10 most common fueling locations and how often they were visted 
def most_common_locs(df):
    print("\nExercise 3:")
    df_location_count = pd.DataFrame(df.location.value_counts())
    df_location_count = df_location_count.reset_index()
    df_location_count.columns = ['location', 'count']
    df_location_count = df_location_count.sort_values(['count','location'], ascending =[False, True])
    for index, row in df_location_count.iterrows():
        if index > 9:
            break
        else:
            print('{} {}'.format(row['location'], row['count']))
        
        
    

#Total visits to each state
def state_totals(df):
    print("\nExercise 4:")
    states = df.location.str[-2:]
    state_count = pd.DataFrame(states.value_counts())
    state_count = state_count.reset_index()
    state_count.columns = ['state', 'count']
    state_count = state_count.sort_values('state')
    for index, row in state_count.iterrows():
        print('{} {}'.format(row['state'], row['count']))
    

    
    #print(state_count.to_string(header = False, index = False))



#Unique dates of fuelings
def unique_dates(df):
   print("\nExercise 5:")
   dates = []
   for i in df['date']:
       if pd.isna(i):
           continue
       else:
           dates.append(i.strftime('%Y/%m/%d'))
   dates_df = pd.DataFrame(dates)
   dates_df.columns = ['dates']
   date_count = dates_df.dates.str[-5:]
   date_count = date_count.value_counts()
   print(date_count.count())



#Average price for gas per month
def month_avg(df):
    print("\nExercise 6:")
    cost = []
    months_num = []
    for index, row in df.iterrows():
        if pd.isna(row['date']) or pd.isna(row['cost']):
            continue
        else:
            cost.append(row['cost'])  
            months_num.append(row['date'])
     
    for i in range(len(months_num)):
        months_num[i] = months_num[i].month

    df2 = pd.DataFrame(months_num, cost)
    df2 = df2.reset_index()
    df2.columns = ['cost', 'month']
    df2 = df2.groupby(['month']).mean().reset_index()
    df2['month'] = df2['month'].apply(lambda x: calendar.month_name[x])
    for index, row in df2.iterrows():
        print('{} ${:.2f}'.format(row['month'], round(row['cost'], 2)))
    


#########################

if __name__ == '__main__':
    data_file_name = sys.argv[1]
    main(data_file_name)




