### Find the continents where all countries have a population <= 25000000. Then find the names of the countries associated with these continents. Show name, continent and population.
SELECT x.name, x.continent, x.population
FROM world AS x
WHERE 25000000 > ALL (
  SELECT y.population
  FROM world AS y
  WHERE x.continent = y.continent);
