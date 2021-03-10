# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:00:16 2021

@author: Jarrod Daniels
"""
from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import urllib.parse
import scraper
import re

birthday = []
name = []
color = []
gender = []
price = []
keepGoing = False


RabbitScraper = scraper.UrlScraper('https://www.bluecloverrabbitry.com/adopt.html#/')
RabbitScraper.move_to('<div class="paragraph">')
RabbitScraper.move_to('Born:')
tempBDay = RabbitScraper.pull_until("</font>")
RabbitScraper.move_to('<br>')
RabbitScraper.move_to('</strong>')
for i in range(2):
    RabbitScraper.move_to('<br />')
textToEdit = RabbitScraper.pull_until("</div>")

    
def extractRabbits(string, birthday_str):
    strBreakUp1 = string.split('<br />')
    for buns in strBreakUp1:
        strBreakUp2 = buns.split("-")
        strBreakUp2[2] =  strBreakUp2[2].strip()
        tempColor = strBreakUp2[1].rsplit(' ', 1)[0]
        tempColor = tempColor.strip()
        brokenWords = strBreakUp2[1].split()
        tempGender = brokenWords[-1]
        birthday_str = birthday_str.strip()
        birthday.append(birthday_str)
        name.append(strBreakUp2[0])
        color.append(tempColor)
        gender.append(tempGender)
        price.append(strBreakUp2[2])
        

def keepExtracting(scraperObj):
    RabbitScraper.move_to('<div class="paragraph">')
    RabbitScraper.move_to('Born:')
    tempBirthDay = RabbitScraper.pull_until("</font>")
    RabbitScraper.move_to('</strong>')
    for i in range(2):
        RabbitScraper.move_to('<br />')
    textToEdit2 = RabbitScraper.pull_until("</div>")
    print(textToEdit2)
    #extractRabbits(textToEdit, tempBirthDay)
    
    

def checkNextP(scraperObj):
    scraperObj.move_to('<div class="paragraph">')
    examineText = scraperObj.pull_until('</div>')
    if "available" not in examineText:
        return False
    else:
        return True
    
    
extractRabbits(textToEdit, tempBDay)
print(name)
print(color)
print(gender)
print(price)
print(birthday)

