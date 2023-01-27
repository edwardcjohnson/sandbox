# build and run an image with the following commands:
# docker build -t python:3.9_pandas
# docker run -it python:3.9_pandas
FROM python:3.9

RUN pip install pandas sqlalchemy psycopg2-binary

WORKDIR /app
COPY pipeline.py pipeline.py

ENTRYPOINT ["python", "pipeline.py"]