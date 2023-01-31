```shell
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    restart: always
```

# Create docker network so that the python process running in one container can talk to postgres running in another container
`docker network create pg-network`

# Run the postgres container
```shell
docker run -it \
  -e POSTGRES_USER="root" \
  -e POSTGRES_PASSWORD="root" \
  -e POSTGRES_DB="ny_taxi"\
  -v $(pwd)/ny_taxi_postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  --network=pg-network \
  --name pg-database \
  postgres:13
```

# Build our data ingestion image
```shell
sudo docker build -t taxi_ingest:v1 .
```

# Run our data ingestion container
```shell
docker run -it \
--network=pg-network \
taxi_ingest:v1
```

# Login to Postgres
`pgcli -h localhost -p 5432 -u root ny_taxi`

## List tables
`\dt`

# Google Cloud Setup (DE Zoomcamp 1.3.1)
https://github.com/DataTalksClub/data-engineering-zoomcamp/tree/main/week_1_basics_n_setup/1_terraform_gcp
## Install gcloud CLI for Ubuntu:
https://cloud.google.com/sdk/docs/downloads-snap

# Terraform
```shell
terraform init
terraform plan
terraform apply
terraform destroy
```