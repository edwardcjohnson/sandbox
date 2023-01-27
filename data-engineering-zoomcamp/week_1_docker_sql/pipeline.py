import pandas as pd
import sqlalchemy
import time

def main():
    start = time.time()

    engine= sqlalchemy.create_engine(
            "postgresql://root:root@pg-database:5432/ny_taxi" #  "postgresql://root:root@localhost:5432/ny_taxi"
            )   
    pd.read_csv(
        "https://github.com/DataTalksClub/nyc-tlc-data/releases/download/green/green_tripdata_2019-01.csv.gz",
        ).to_sql("green_tripdata", con=engine, chunksize=10000, if_exists="replace")

    pd.read_csv(
        "https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv",
        ).to_sql("taxi_zone_lookup", con=engine, chunksize=10000, if_exists="replace") 
                                                                                            
    end = time.time()
    print(f"Job finished and took {end - start} seconds to run")


if __name__== '__main__':
    main()