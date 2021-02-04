#!/usr/bin/python
import sys
import csv
import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
  
# Python Analytics Practice

#Executes all functions in the program
def main(file_name):
    rows = read_data(file_name)
    total_cost(rows)
    least_common_locs(rows)
    most_common_locs(rows)
    state_totals(rows)
    unique_dates(rows)
    month_avg(rows)
    highest_thirty(rows)
    plot_monthly(rows)
    plot_mpg(rows)
    plot_betweens(rows)


# Reading in data    
def read_data(file_name):
    rows = []
    with open(file_name) as csvfile:
        read = csv.reader(csvfile)
        next(read)
        for r in read:
             if r[0] == '' or r[1] == '' or r[2] == '' or r[3] == '' or r[4] == '':
                 continue
             else:
                 r[0] = datetime.datetime.strptime(r[0], '%m/%d/%Y').date()
                 r[1] = int(r[1])
                 r[3] = float(r[3])
                 r[4] = float(r[4].replace('$', ''))
             rows.append((r[0], r[1], r[2], r[3], r[4]))
    return rows



# Prints out the total amount of money spent on gas.
def total_cost(rows):
    print("\nExercise 1:")
    amt_spent = [r[3]*r[4] for r in rows if r[3] and r[4]]
    total_cost = sum(amt_spent)
    print("$"+"{}".format(total_cost))
    



# Print out the number of refueling locations that were visited exactly once. 
def least_common_locs(rows):
    print("\nExercise 2:")
    all_locations = [r[2] for r in rows if r[2]]
    unique_refueling_locations = []
    for l in all_locations:
        comparing_string = l
        location_count = all_locations.count(comparing_string)
        if location_count == 1:
            unique_refueling_locations.append(comparing_string)
    
    print(len(unique_refueling_locations))
        

# Prints out the 10 most common refueling locations, along with the number of times they
# appear in the data, in descending order.
def most_common_locs(rows):
    print("\nExercise 3:")
    all_locations = [r[2] for r in rows if r[2]]
    location_count_dict = {l : all_locations.count(l) for l in all_locations }
    final_locations = list(location_count_dict.items())
    final_locations = sorted(final_locations, key =lambda x: x[1], reverse = True)
    for j,d in final_locations[0:10]:
        print(j + " " + str(d))
    
    

# Prints out the total number of visits for each state (as designated by the two-letter
# abbreviation at the end of the location string, one per line, in alphabetical order:
def state_totals(rows):
    print("\nExercise 4:")
    all_locations = [r[2] for r in rows if r[2]]
    all_state_locations = [r[-2:] for r in all_locations]
    all_state_locations.sort()
    state_count = {l : all_state_locations.count(l) for l in all_state_locations }
    for i in state_count:
        print("{} {}".format(i,state_count[i]))
    
    

# Prints out the total number unique dates in the calendar year that refueling took place
def unique_dates(rows):
    print("\nExercise 5:")
    dates = [r[0].strftime('%Y%m%d') for r in rows if r[0]]
    dates = [r[-4:] for r in dates]
    unique_dates = []
    for d in dates:
        if d not in unique_dates:
            unique_dates.append(d)
    unique_num = len(unique_dates)
    print(unique_num)
    
    
# Prints out the average price per gallon for each month of the year, in calendar order, like so:
def month_avg(rows):
    print("\nExercise 6:")
    dates = [r[0].strftime('%m') for r in rows if r[0] and r[4]]
    dates = [int(d) for d in dates]
    prices = [r[4] for r in rows if r[0] and r[4]]
    date_prices = list(zip(dates,prices))
    date_prices = sorted(date_prices, key=lambda x: x[0])
    dictionary = {}
    
    for k, v in date_prices:
        dictionary.setdefault(k, []).append(v)
    
    avgDict = {}
    for k, v in dictionary.items():
        avgDict[k] = sum(v)/ float(len(v))
    
    for k in avgDict:
        print("{} ${}".format(calendar.month_name[k], round(avgDict[k], 2)))


# Print out the start and end dates for top three periods with the most miles driven in thirty
# days or less. (periods do not overlap with each other)             
def highest_thirty(rows):
    print("\nExercise 7:")
    dates = [r[0] for r in rows]
    miles = [r[1] for r in rows]
    miles_reversed = list(reversed(miles))
    dates_reversed = list(reversed(dates))
    date_miles = list(zip(dates, miles))
    date_miles2 = list(zip(dates_reversed, miles_reversed))
    new_list = []
    miles_traveled = []

   
    for i in date_miles:
        for j in date_miles2:
            if (j[0]-i[0]).days <= 30:
                new_list.append((i,j))
                break
   
   
    for i in new_list:
        miles_traveled.append((i[0][0], i[1][0], (i[1][1] - i [0][1])))
   
    miles_traveled = sorted(miles_traveled, key =lambda x: x[2], reverse = True)
   
    date_num_list = []
    for i in miles_traveled:
        date_num_list.append((int(i[0].strftime("%Y%m%d")), int(i[1].strftime("%Y%m%d")), i[2]))
   
    temp = [date_num_list[0][0], date_num_list[0][1]]
    unique_list1 = []
    
    for i in date_num_list[1:]:
        if (temp[0] <= i[0] <= temp[1]) or (temp[0] <= i[1] <= temp[1]):
            continue
        else:
            unique_list1.append(i)
    
    temp2 = [unique_list1[0][0], unique_list1[0][1]]
    unique_list2 = []
    
    for i in unique_list1[1:]:
        if (temp2[0] <= i[0] <= temp2[1]) or (temp2[0] <= i[1] <= temp2[1]):
            continue
        else:
            unique_list2.append(i)
            
    final_date_list = []
    final_date_list.append((date_num_list[0]))
    final_date_list.append((unique_list1[0]))
    final_date_list.append((unique_list2[0]))
    
    
    list1 = []
    list1.append(final_date_list[0][0])
    list1.append(final_date_list[0][1])
    list1.append(final_date_list[0][2])
    list2 = []
    list2.append(final_date_list[1][0])
    list2.append(final_date_list[1][1])
    list2.append(final_date_list[1][2])
    list3 = []
    list3.append(final_date_list[2][0])
    list3.append(final_date_list[2][1])
    list3.append(final_date_list[2][2])
    
    for i in range(len(list1)):
        list1[i] = str(list1[i])
        list2[i] = str(list2[i])
        list3[i] = str(list3[i])
    
    list1[0] = datetime.datetime(year=int(list1[0][0:4]), month=int(list1[0][4:6]), day=int(list1[0][6:8]))
    list1[1] = datetime.datetime(year=int(list1[1][0:4]), month=int(list1[1][4:6]), day=int(list1[1][6:8]))
    list2[0] = datetime.datetime(year=int(list2[0][0:4]), month=int(list2[0][4:6]), day=int(list2[0][6:8]))
    list2[1] = datetime.datetime(year=int(list2[1][0:4]), month=int(list2[1][4:6]), day=int(list2[1][6:8]))
    list3[0] = datetime.datetime(year=int(list3[0][0:4]), month=int(list3[0][4:6]), day=int(list3[0][6:8]))
    list3[1] = datetime.datetime(year=int(list3[1][0:4]), month=int(list3[1][4:6]), day=int(list3[1][6:8]))
    list1[0] = list1[0].strftime("%Y-%m-%d")
    list1[1] = list1[1].strftime("%Y-%m-%d")
    list2[0] = list2[0].strftime("%Y-%m-%d")
    list2[1] = list2[1].strftime("%Y-%m-%d")
    list3[0] = list3[0].strftime("%Y-%m-%d")
    list3[1] = list3[1].strftime("%Y-%m-%d")
    
    print(list1[0]+" "+list1[1]+" "+list1[2]+" miles")
    print(list2[0]+" "+list2[1]+" "+list2[2]+" miles")
    print(list3[0]+" "+list3[1]+" "+list3[2]+" miles")



# creates a bar chart in matplotlib that indicates the total number of miles driven in each month
def plot_monthly(rows):
   print("\nExercise 8:")
   dates = [r[0].strftime("%Y%m") for r in rows]
   dates = [int(d) for d in dates]
   mile_count = [r[1] for r in rows]
   date_miles = list(zip(dates,mile_count))
   driven = []
   for i in range(len(date_miles)-1):
       driven.append(mile_count[i+1]-mile_count[i])
   del dates[-1]
   date_miles2 = list(zip(dates, driven))
   dictionary = {}
   for k, v in date_miles2:
       dictionary.setdefault(k, []).append(v)
   ranged = {}
   for k, v in dictionary.items():
       ranged[k] = sum(v)
   
   dict_list = []
   for k, v in ranged.items():
       temp = [k, v]
       dict_list.append(temp)
   for i in dict_list:
       i[0] = str(i[0])
       i[0] = i[0][-2:]
       i[0] = int(i[0])
   final_dict = {}
   for a, b in dict_list:
        final_dict.setdefault(a, []).append(b)
   final_dict2 = {}
   for k, v in final_dict.items():
       final_dict2[k] = sum(v)
   final_dict2 = sorted(final_dict2.items(), key=lambda x: x[0])
   final_miles = []
   for j, k in final_dict2:
       final_miles.append(k)

   months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September','October','November', 'December']
   
   x_pos = [i for i, _ in enumerate(months)]
   plt.figure(figsize=(13,13))
   plt.bar(x_pos, final_miles, color=(0.4, 0.2, 0.6, 0.6))
   plt.xlabel("Month")
   plt.ylabel("Miles Driven")
   plt.title = ("Total Miles Driven by Month")
   plt.xticks(x_pos, months)
   #plt.savefig("mustard_months.pdf", bbox_inches = "tight")
   plt.show()
   
   
# Use matplotlib to create a line plot of the miles per gallon achieved (y-axis) over time
def plot_mpg(rows):
    print("\nExercise 9:")
    dates3 = [r[0] for r in rows]
    miles = [r[1] for r in rows]
    gallons = [r[3] for r in rows]
    miles_traveled = [miles[i + 1] - miles[i] for i in range(len(miles)-1)]
    refule_dates = []
    for i in dates3[:-1]:
        refule_dates.append(i)
    final_gals = []
    for i in gallons [:-1]:
        final_gals.append(i)    
    mpg = []
    for m, g in zip(miles_traveled, final_gals):
        mpg.append(m/g)
    
    plt.figure(figsize =(13,13))
    plt.xlabel("Time")
    plt.ylabel("miles per gallon")
    plt.plot(refule_dates, mpg)
    #plt.savefig("mustard_mpg_time.pdf", bbox_inches = "tight")
    plt.show()



# creates a density plot of the number of days between refueling entries over the period
# captured in the data.
#
def plot_betweens(rows):
    print("\nExercise 10:")
    dates = [r[0] for r in rows]
    dates_between = [dates[i + 1] - dates[i] for i in range(len(dates)-1)]
    final_list = []
    for i in dates_between:
        final_list.append(i.days)
    plt.xlabel("Days Between Refills")
    plt.ylabel("Density")
    sns.kdeplot(final_list, shade=True)
    #plt.savefig("mustard_in_between_days.pdf", bbox_inches = "tight")
    plt.show()
    

#########################

if __name__ == '__main__':
    data_file_name = sys.argv[1]
    main(data_file_name)




