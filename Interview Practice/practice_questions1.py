# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:22:11 2021

@author: Jarrod Daniels
"""
import string

### takes a single integer as input and returns the sum of the integers from zero to the input parameter.

def add_it_up(int):
    total = 0
    num_list = list(range(1, int+1))
    for num in num_list:
        total += num
    print(total)
    return total

add_it_up(5)

"""
Creates a basic ceasar cypher that takes in a lower case string and returns the string shifted N times
"""

def c_cypher(str, n):
    return_string = ''
    alphabet_string = string.ascii_lowercase
    alphabet_list = list(alphabet_string)
    for letter in str:
        if letter == ' ':
            return_string += ' '
        else:
            index_to_ref = alphabet_list.index(letter)+4
            if index_to_ref > 26:
                index_to_ref -= 26
            return_string += alphabet_list[index_to_ref]
                
    print(return_string)
    return(return_string)

example_string = "holland lopz"
n = 5
c_cypher(example_string, n)


# Other implementation using minimal functions
def caesar(plain_text, shift_num):
    letters = string.ascii_lowercase
    mask = letters[shift_num:] + letters[:shift_num]
    trantab = str.maketrans(letters, mask)
    print(plain_text.translate(trantab))

caesar(example_string, 4)

