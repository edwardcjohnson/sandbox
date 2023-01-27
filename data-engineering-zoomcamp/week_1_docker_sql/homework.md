# Question 1. Knowing docker tags

--iidfile string

# Question 2. Unstanding docker first run

3 python packages

# Question 3. Count Records
select count(*) FROM green_tripdata where lpep_pickup_datetime like '%-01-15 %';  

# Question 4.
select lpep_pickup_datetime FROM green_tripdata where trip_distance = (select MAX(trip_distance) FROM green_tripdata);                                                


# Question 5.
select passenger_count, count(*) FROM green_tripdata where lpep_pickup_datetime like '2019-01-01 %' GROUP BY passenger_count HAVING passenger_count<4;                


# Question 6.
select "Zone" from taxi_zone_lookup where "LocationID"=(select "DOLocationID" from green_tripdata where fare_amount =  (select MAX(fare_amount) FROM green_tripdata jo
 in taxi_zone_lookup on "PULocationID" = "LocationID" where "Zone"='Astoria')); 