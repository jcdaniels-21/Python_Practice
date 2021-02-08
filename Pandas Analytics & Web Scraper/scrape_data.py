import sys
import urllib.parse
import scraper
import pandas as pd
import csv

countries = []
population = []
percent_of_world_pop = []


wiki_scrape = scraper.UrlScraper('https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population')
wiki_scrape.move_to('<td>1')
i = 1
while i <= 240:
    wiki_scrape.move_to('title="')
    countries.append(wiki_scrape.pull_until('"'))
    wiki_scrape.move_to('</td>')
    population.append(wiki_scrape.pull_from_to('<td>', '</td>'))
    wiki_scrape.move_to('</td>')
    percent_of_world_pop.append(wiki_scrape.pull_from_to('<td>', '</td>'))
    i+=1

for i in range(len(countries)):
    population[i] = population[i].strip()
    percent_of_world_pop[i] = percent_of_world_pop[i].strip()
    if "span" in percent_of_world_pop[i]:
      percent_of_world_pop[i] = percent_of_world_pop[i][73:]
    
  
df = pd.DataFrame()
df['Country/Dependency'] = countries
df['Population Size'] = population
df['Percent of World Population'] = percent_of_world_pop



with open('awesome_data.csv', 'w', newline = '', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Country/Dependency', 'Population Size', 'Percent of World Population'])
    for i in df.index:
        writer.writerow([df['Country/Dependency'][i],df['Population Size'][i],df['Percent of World Population'][i]])

## The output of this web scrape can be found in the 'awesome_data.csv' file which is contained in this folder