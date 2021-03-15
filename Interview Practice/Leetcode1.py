# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:05:18 2021

@author: Jarrod Daniels
"""

example = [0, 2, 3, 5, 8]
example2 = [0, 8, 6, 10, 11, -1]
example3 = [3,3]
target_num = 7
pali_list = [1, 2, 2, 1]

def twoSum(num, target):
    for i in range(len(num)):
        for j in range(len(num)):
            if num[i] == num[j]:
                continue
            else:
                sum = num[i] + num[j]
            if sum == target:
                return_list = [i, j]
                print(return_list)
                return return_list

twoSum(example, target_num)

"""
Second version deals with issue where you might have two of the same number in the array
"""

def twoSum2(nums, target):
        d = {}
        ans = []
        for i in range(len(nums)):
            if target-nums[i] in d:
                ans.append(i)
                ans.append(d[target-nums[i]])
            else:
                d[nums[i]] = i
        return ans
    
twoSum(example2, target_num)

"""
Finds out if the array is a palindrome
"""

def isPalindrome(head):
    isPal = True
    reversedList = []
    for i in reversed(head):
        reversedList.append(i)
    for j in range(len(head)):
        if head[i] == reversedList[i]:
            continue
        else:
            isPal = False
            break
    print(isPal)
    return isPal

"""
Finds the running sum of an array [1, 1, 1, 1] = [1, 2, 3, 4]
"""

def runningSum(nums):
        fin_arr = []
        for i in range(len(nums)):
            if i == 0:
                fin_arr.append(nums[i])
            else:
                fin_arr.append(nums[i] + fin_arr[i-1])
        return fin_arr
    
"""
finds pairs where first index is less than the second
"""
def numIdenticalPairs(self, nums):
        return (sum(i==j for j in nums for i in nums)-len(nums))//2    
"""
def numIdenticalPairs(self, nums):
        goodPairs = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                if(i >= j):
                    continue
                else:
                    if(nums[i] == nums[j]):
                        goodPairs+=1
                        print(i, j)
        return goodPairs
"""



isPalindrome(example)
isPalindrome(pali_list)

