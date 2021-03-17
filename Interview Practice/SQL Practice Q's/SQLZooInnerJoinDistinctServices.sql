#Give a list of all the services which connect stops 115 and 137 ('Haymarket' and 'Leith')

SELECT DISTINCT a.company, a.num
FROM route AS a 
JOIN route AS b ON (a.company = b.company AND a.num = b.num)
WHERE a.stop = 115 and b.stop = 137