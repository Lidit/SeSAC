--SELECT Name || '음악' as Name FROM genres;
--SELECT DISTINCT Country FROM customers;
--SELECT Milliseconds / 1000 as Seconds  from tracks;

SELECT Round(Milliseconds / 1000/60) as minutes FROM tracks;

SELECT Total ,
CASE
	WHEN Total >50 THEN '높음'
	WHEN (Total <=50) & (Total >10) THEN '보통'
	WHEN Total <=10 THEN '낮음'
END as 'Category'
FROM invoices  


SELECT *
FROM customers
WHERE Country == "USA";

SELECT *
FROM employees
WHERE HireDate < "2003-01-01";

SELECT *
FROM albums
WHERE Title LIKE "%Love%";

-- Groupby 2, 4

SELECT  GenreId , COUNT(TrackId) as TrackCount
FROM tracks 
Group By GenreId  
HAVING COUNT(TrackId) >= 100;

SELECT CustomerId, SUM(Total) 
FROM invoices 
Group By CustomerId
HAVING SUM(Total) >= 40;


SELECT *
FROM artists 
ORDER BY Name ASC;

SELECT *
FROM tracks
ORDER BY Milliseconds ;

SELECT *
FROM invoices
ORDER BY Total DESC;



SELECT *
FROM invoices as i
JOIN customers as c ON
	i.CustomerId = c.CustomerId;
	
SELECT i.InvoiceId, ii.Quantity 
FROM invoice_items ii 
LEFT JOIN invoices i ON
	ii.InvoiceId = i.InvoiceId ;

-- 1. customers와 invoices을 테이블을 조인하여 각 고객의 이름과 구매한 청구서의 총 금액을 선택
SELECT c.FirstName || " " || c.LastName ,SUM(Total)
FROM invoices i 
JOIN customers c ON
	i.CustomerId = c.CustomerId;

SELECT c.FirstName || " " || c.LastName as FullName, SUM(i.Total) as TotalPurchase
FROM invoices i 
LEFT JOIN customers c ON
	i.CustomerId = c.CustomerId
GROUP BY c.CustomerId;