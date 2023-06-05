# Preparing the training data
Generate a synthetic dataset with 1000 samples and save it as `data.csv` with default feature ranges of `0-1` for each feature:<br>
`python prep_data.py -n 1000 -r 0 1 0 1 0 1 -o data.csv`
## Running the data prep from a Docker container
To build the Dockerfile.prep, navigate to the directory containing the Dockerfile.prep file in your terminal, then run the following command:
```
docker build -t prep_data-image -f Dockerfile.prep_data .
```
This command builds a Docker image with the tag prep_data-image and uses the Dockerfile.prep file as the build context.
This command mounts the current working directory to the `/app/data directory` in the container and writes the output file to the mounted directory. The output file should now be available in your local filesystem in the same directory where you ran the docker run command.
```
docker run -v $(pwd)/data:/app/data prep_data-image python prep_data.py -n 1000 -r 0 1 0 1 0 1 -o /app/data/data.csv
```
This will run the `prep_data.py` script inside the container, passing the arguments `-n 1000 -r 0 1 0 1 0 1 -o data.csv` to it, saving the data to the mounted dir `$(pwd)/data`.

# Training the model
Train and save the model to a file named `model.txt` by running:
`python train_model.py --dataset_path=data.csv --model_name=model.txt`
## Running the model training from a Docker container

To build the Docker image using the Dockerfile named "Dockerfile.train", navigate to the directory containing the Dockerfile in a terminal window and run the following command:
```
docker build -t train-image -f Dockerfile.train .
```
This will build a Docker image tagged as "train-image" using the specified Dockerfile. The `.` at the end of the command specifies that the build context is the current directory.
You can override any of the command line arguments of a Docker container during the docker run command using the --entrypoint option. You can mount the current working directory to the `/data` directory inside the container and then run the container. This assumes that the `data.csv` file is located in `$(pwd)/data`. You can modify the path to the file as needed. For example:
```
docker run -e MODEL_NAME=model.txt -e DATASET_PATH=/data/data.csv -v $(pwd)/data:/data -v $(pwd)/models:/app/models train-image python /app/train_model.py --dataset_path=/data/data.csv --model_name=/app/models/model.txt
```
This command mounts the data directory to `/data` in the container and mounts the models directory to `/app/models` in the container. It also runs the `train_model.py` script located at `/app/train_model.py` in the container and saves the model to `/app/models/model.txt`.

# Running the prediction app
Start the FastAPI app by running `uvicorn predict:app --reload`. This will start the app and listen for incoming requests at `http://localhost:8000`. You can test the API by sending a POST request to `http://localhost:8000/predict` with a JSON payload containing values for the `feature1`, `feature2`, and `feature3` fields.

The following will send a GET request to your endpoint and return the response:<br>
`curl http://localhost:8000/predict`

Send a POST request instead and include the required data in the request body.
Our endpoint requires a JSON payload containing the data to be predicted, so use the following command to send a POST request with the required data:<br>
`curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"feature1": 0.5, "feature2": 0.7, "feature3": 0.2}'`

Or using the `payload.json` file that is available:<br>
`curl -X POST -H "Content-Type: application/json" -d @payload.json http://localhost:8000/predict`

This should return a prediction that resembles:<br>
`{"prediction":0.9130092931232355}`

## Running the prediction app from a Docker container
To run the app in a container, we need to first create a Dockerfile that defines the container image.
The `Dockerfile` defines an image based on Python 3.9 and installs the necessary dependencies.
Here is a brief explanation of each line:

`FROM python:3.9`: This specifies the base image for the container.<br>
`WORKDIR /app`: This sets the working directory to /app.<br>
`COPY requirements.txt requirements.txt`: This copies the requirements.txt file from the host machine to the container.<br>
`RUN pip3 install -r requirements.txt`: This installs the dependencies listed in requirements.txt.<br>
`COPY . .`: This copies the current directory (which contains the predict.py file and the model.txt file) to the container.<br>
`CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "80"]`: This specifies the command that should be run when the container starts. In this case, it starts the Uvicorn server that serves the FastAPI app on port 80.

To build the Docker image with the tag `predict-image`, run the following command in the same directory as the Dockerfile:<br>
`docker build -t predict-image -f Dockerfile.predict .`<br>

To start a container based on the `predict-image` image and map port `80` in the container to port `8080` on the host machine, use the following command:<br>
`docker run -p 8080:80 -v $(pwd)/models:/models -e MODEL_FILE_PATH=/models/model.txt predict-image`

Access the docs endpoint by visiting `http://localhost:8080/docs` in your web browser.

Test the predict endpoint with:<br>
```
curl -X POST "http://localhost:8080/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"feature1": 0.5, "feature2": 0.7, "feature3": 0.2}'
```

# Deploying the prediction app on Kubernetes
## Build images
Follow the previous steps to ensure that the model object has been created, and the predict-image image has been built.

## Installation
Minikube start guide:<br>
https://minikube.sigs.k8s.io/docs/start/
We will need to mount the model directory when we start minikube:<br>
`minikube start --mount-string="$HOME/projects/sandbox/model_api/models:/models  --mount=True`
We will need to push images into minikube like this:<br>
`minikube image load <image name>`
If your image changes after your cached it, run:
`minikube cache reload`
For more details on images in minikube, reference: 
https://minikube.sigs.k8s.io/docs/handbook/pushing/#2-push-images-using-cache-command

Helpful commands:
Reference: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
```
kubectl get services
minikube service <service name> # Get the service's info
minikube service <service-name> --url # Get the service's URL
kubectl delete service <service name>
kubectl delete deployment <deployment name>
```

## Running the app
Apply the service and deployment manifests using the following commands:
```
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/deployment.yaml
```

```
curl -X POST $(minikube ip):30001/predict -H "accept: application/json" -H "Content-Type: application/json" -d '{"feature1": 0.5, "feature2": 0.7, "feature3": 0.2}'
```

## Troubleshooting deployment image helpful commands
```
minikube delete # delete the minikube container to modify the mount
minikube start --mount-string="$HOME/projects/sandbox/model_api/models:/src"  --mount=True
minikube ssh # connect to the minikube container to verify that the mount worked
ls -alh /models

# remove the image before editing and rebuilding
docker image rm predict-image
docker build -t predict-image -f Dockerfile.predict . --no-cache
minikube image load predict-image

kubectl delete deployment predict-deployment
kubectl apply -f kubernetes/deployment.yaml

kubectl delete deployment predict-service
kubectl apply -f kubernetes/service.yaml
```

# Scaling the app
Here are a few options to scale a FastAPI app, depending on your requirements and resources:

Use a reverse proxy/load balancer: You can use a reverse proxy such as Nginx or a load balancer such as HAProxy to distribute incoming requests across multiple instances of your FastAPI app running on different servers. This can help distribute the load and improve the availability of your app.

Use a container orchestrator: You can use a container orchestrator such as Kubernetes or Docker Swarm to manage and scale your app. With a container orchestrator, you can define the number of replicas or instances of your app that should be running and the orchestrator will handle the deployment, scaling, and load balancing of your app.

Use a cloud provider: Many cloud providers such as AWS, GCP, and Azure offer services for running and scaling containerized applications, such as ECS, EKS, GKE, and AKS. These services provide a managed environment for running your app and can scale automatically based on demand.

Optimize the app: You can optimize your FastAPI app to handle more requests with the same resources. Some techniques include using asynchronous code, caching responses, and optimizing database queries.

Keep in mind that scaling an app can be complex and requires careful planning and monitoring. You should also consider the cost, complexity, and performance tradeoffs of different scaling strategies.
