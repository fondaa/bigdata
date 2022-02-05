libname chinook "C:\Users\ffan\OneDrive - IESEG\Business Reporting Tools\Chinook dataset-20201021";
/*Financial
 company sales evolution over the years */
proc sql;
select distinct year(datepart(invoicedate)) as Year, sum(total) as Sales
from chinook.invoices
group by year(datepart(invoicedate));
quit;
/*purchases made per invoice*/
proc sql;
select invoiceID as Invoice, sum(quantity) as Nbr_Purchase
from chinook.invoice_items
group by invoiceID;
quit;
/*average purchases made per invoice*/
proc sql;
select avg(Nbr_Purchase)
from
(select invoiceID as Invoice, sum(quantity) as Nbr_Purchase
from chinook.invoice_items
group by invoiceID);
quit;

/*- Customers. We don’t want an overview per customer as this would be too long.
Think about showing averages and/or for instance exceptional (bad or good)
customers.

/*How many customers do we have + Are they company clients or not*/
proc sql;
select "CompanyClients", count(customerID) as NbrCustomers
from chinook.customers
where company <> "NA"
union
select "NonCompanyClients", count(customerID) 
from chinook.customers
where company = "NA";
quit;
/*How long has it been since the last customer purchase (recency)*/
proc sql;
select yrdif(max(datepart(invoicedate)), today()) as Yr_Since_Last_Purchase
from chinook.invoices;
quit;
/* average number of purchase done by customer*/
proc sql;
select avg(frequency) as Nbr_of_Purchase_Average_Customer
from
(select c.customerID, count(i.total) as frequency
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
group by c.customerID);
quit;
/* average monetary value spent by customer*/
proc sql;
select avg(monetary) as PurchaseAmount_Average_Customer
from
(select c.customerID, sum(i.total) as monetary
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
group by c.customerID);
quit;
/*average customer tenure with company*/
proc sql;
select avg(CustomerTenure_Years)
from
(select customerID, yrdif(min(datepart(invoicedate)),max(datepart(invoicedate))) as CustomerTenure_Years
from chinook.invoices
group by customerID);
quit;
/*sales by country */
proc sql;
select c.country, sum(i.total) as SalesPerCountry
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
group by c.country;
quit;
/*sales per market region*/
proc sql;
select "Sales_NorthAmerica", sum(i.total) as Sales_Per_Market
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("USA", "Canada")
union
select "Sales_SouthAmerica", sum(i.total)
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("Brazil", "Argentina", "Chile")
union
select "Sales_APAC", sum(i.total) 
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("Australia", "India")
union
select "Sales_Europe", sum(i.total) 
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("Germany", "Norway","Czech Republic","Austria","Belgium","Denmark","Portugal","France","Finland",
"Hungary","Ireland","Italy","Netherlands","Poland","Spain","Sweden","United Kingdom")
order by 2 desc;
quit;
/* the top-listened-to artists for each of the 4 market regions */

PROC SQL;
SELECT "NorthAmerica", r.Name as TopArtist, count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("USA", "Canada")
GROUP BY r.Name
having TotalNumber >= all
(SELECT count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("USA", "Canada")
GROUP BY r.Name)
union
SELECT "SouthAmerica", r.Name, count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("Brazil", "Argentina", "Chile")
GROUP BY r.Name
having TotalNumber >= all
(SELECT count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("Brazil", "Argentina", "Chile")
GROUP BY r.Name)
union
SELECT "APAC", r.Name, count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("Australia", "India")
GROUP BY r.Name
having TotalNumber >= all
(SELECT count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("Australia", "India")
GROUP BY r.Name)
union
SELECT "Europe", r.Name, count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("Germany", "Norway","Czech Republic","Austria","Belgium","Denmark","Portugal","France","Finland",
"Hungary","Ireland","Italy","Netherlands","Poland","Spain","Sweden","United Kingdom")
GROUP BY r.Name
having TotalNumber >= all
(SELECT count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("Germany", "Norway","Czech Republic","Austria","Belgium","Denmark","Portugal","France","Finland",
"Hungary","Ireland","Italy","Netherlands","Poland","Spain","Sweden","United Kingdom")
GROUP BY r.Name);
QUIT;

/*How is the artist "IronMaiden" ranked in South America?*/

PROC SQL outobs = 15;
SELECT "SouthAmerica", r.Name as ArtistRanking, count(*) as TotalNumber
FROM chinook.customers c 
	 INNER JOIN chinook.invoices as i 
	 on c.customerID = i.customerID 
	 INNER JOIN chinook.invoice_items as ii 
	 on i.invoiceID = ii.invoiceID 
	 INNER JOIN chinook.tracks as t
	 on t.trackid = ii.trackid 
	 INNER JOIN chinook.albums as a
	 on t.albumID = a.albumID
	 inner join chinook.artists as r
	 on a.artistID = r.artistID
WHERE country in ("Brazil", "Argentina", "Chile")
GROUP BY 2
order by 3 desc;
QUIT;

/*Internal business processes:


/*number of genres*/
proc sql;
select distinct count(*)
from chinook.genres;
quit;
/*genres that are bought most/least */
proc sql;
select g.name as Genre, sum(z.quantity) as Nbr_Bought
from chinook.genres as g
	 left join chinook.tracks as t
	 on g.genreID = t.genreID
	 left join chinook.invoice_items as z
	 on t.trackID = z.trackID	 
group by 1
order by 2 desc;
quit;
/*tracks that are bought most/least */
proc sql;
select t.name as Tracks, sum(z.quantity) as NbrBought
from chinook.tracks as t
	left join chinook.invoice_items as z
	on t.trackID = z.trackID
	left join chinook.invoices as i
	on z.invoiceID = i.invoiceID
group by 1
order by 2 desc;
quit;
/* top 6 popular tracks bought*/
proc sql outobs = 6;
select t.name as Tracks, sum(z.quantity) as NbrBought
from chinook.tracks as t
	left join chinook.invoice_items as z
	on t.trackID = z.trackID
	left join chinook.invoices as i
	on z.invoiceID = i.invoiceID
group by 1
order by 2 desc;
quit;
/*mediatype that are bought most/least */
proc sql;
select m.name as MediaType, sum(z.quantity) as Nbr_Bought
from chinook.media_types as m
	 left join chinook.tracks as t
	 on m.mediatypeID = t.mediatypeID
	 left join chinook.invoice_items as z
	 on t.trackID = z.trackID	 
group by 1
order by 2 desc;
quit;
/* number of tracks that have no sales */
proc sql;
select count(*)
from
(select t.trackID as Tracks, count(i.customerID) as Nbr_Customers
from chinook.tracks as t
	left join chinook.invoice_items as z
	on t.trackID = z.trackID
	left join chinook.invoices as i
	on z.invoiceID = i.invoiceID
group by 1
having Nbr_Customers = 0);
quit;
/*characteristics related to these tracks that have no sales*/
proc sql;
select t.trackID, g.name as GenreName, m.name as MediaTypeName, t.bytes, count(distinct i.customerID) as Nbr_Customers
from chinook.tracks as t
	left join chinook.genres as g
	on t.genreID = g.genreID
	left join chinook.media_types as m
	on t.mediatypeID = m.mediatypeID
	left join chinook.invoice_items as z
	on t.trackID = z.trackID
	left join chinook.invoices as i
	on z.invoiceID = i.invoiceID
group by t.trackID
having Nbr_Customers = 0
order by 2, 3;
quit;
/*how many bytes do we save by deleting these tracks that have no sales*/
proc sql;
select sum(t.bytes) as Total_Bytes_Saved
from
(select t.trackID, g.name as GenreName, m.name as MediaTypeName, t.bytes, count(distinct i.customerID) as Nbr_Customers
from chinook.tracks as t
	left join chinook.genres as g
	on t.genreID = g.genreID
	left join chinook.media_types as m
	on t.mediatypeID = m.mediatypeID
	left join chinook.invoice_items as z
	on t.trackID = z.trackID
	left join chinook.invoices as i
	on z.invoiceID = i.invoiceID
group by t.trackID
having Nbr_Customers = 0);
quit;
/*Employees


/*How many employees do we have*/
proc sql;
select count(*) as Total_Nbr_Employees
from chinook.employees;
quit;
/*employees that are about to retire*/
proc sql;
select employeeID, yrdif(datepart(birthdate),today()) as Age_AboutToRetire
from chinook.employees
where yrdif(datepart(birthdate),today()) > 60;
quit;
/*average employee tenure with the company*/
proc sql;
select avg(EmployeeTenure)
from
(select employeeID, yrdif(datepart(hiredate),today()) as EmployeeTenure
from chinook.employees);
quit;
/*list of employee tenure by employeeIDs*/
proc sql;
select employeeID, yrdif(datepart(hiredate),today()) as EmployeeTenure
from chinook.employees
order by 2;
quit;
/*number of roles at the company*/
proc sql;
select count(distinct title) as Nbr_Roles
from chinook.employees;
quit;
/*How many sales does each of the sales agent have */
proc sql;
select c.supportrepID, count(i.total) as Nbr_of_Sales
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
group by c.supportrepID;
quit;
/*How many sales does each of the supervisors have */
proc sql;
select e.employeeID, count(i.total)
from chinook.customers as c,
	 chinook.invoices as i,
	 chinook.employees as e
where c.customerID = i.customerID
and c.supportrepID = e.employeeID
and lowcase(e.title) = "sales manager"
group by e.employeeID;
quit;
/*sales per agent per country*/
proc sql;
select c.supportrepID, c.country, count(i.total)
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
group by 1,2
order by 1,2;
quit;
/*comparison of number of sales done by each sales agent per market region*/
proc sql;
select "Sales_NorthAmerica", c.supportrepID, count(i.total) as Nbr_Sales
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("USA", "Canada")
group by c.supportrepID

union
select "Sales_SouthAmerica", c.supportrepID, count(i.total) 
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("Brazil", "Argentina", "Chile")
group by c.supportrepID

union
select "Sales_APAC", c.supportrepID, count(i.total) 
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("Australia", "India")
group by c.supportrepID

union
select "Sales_Europe", c.supportrepID, count(i.total) 
from chinook.customers as c,
	 chinook.invoices as i
where c.customerID = i.customerID
and c.country in ("Germany", "Norway","Czech Republic","Austria","Belgium","Denmark","Portugal","France","Finland",
"Hungary","Ireland","Italy","Netherlands","Poland","Spain","Sweden","United Kingdom")
group by c.supportrepID
order by 1 , 3 desc;
quit;

