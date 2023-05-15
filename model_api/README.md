# Running the app
To run the application, first train the model by running `python train_model.py`. This will save the trained model to a file named `model.txt`.

Then, start the FastAPI app by running uvicorn predict:app --reload. This will start the app and listen for incoming requests at http://localhost:8000. You can test the API by sending a POST request to http://localhost:8000/predict with a JSON payload containing values for the feature1, feature2, and feature3 fields.

The following will send a GET request to your endpoint and return the response.
`curl http://localhost:8000/predict`

Send a POST request instead and include the required data in the request body.
For example, if your endpoint requires a JSON payload containing the data to be predicted, you can use the following command to send a POST request with the required data:
`curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"feature1": 0.5, "feature2": 0.7, "feature3": 0.2}'`

Or using the `payload.json` file that is available:
`curl -X POST -H "Content-Type: application/json" -d @payload.json http://localhost:8000/predict`

This should return a prediction that resembles:
`{"prediction":0.9130092931232355}`

# Running the app from a Docker container
To run the app in a container, we need to first create a Dockerfile that defines the container image.
The `Dockerfile` defines an image based on Python 3.9 and installs the necessary dependencies.
Here is a brief explanation of each line:

`FROM python:3.9`: This specifies the base image for the container.
`WORKDIR /app`: This sets the working directory to /app.
`COPY requirements.txt requirements.txt`: This copies the requirements.txt file from the host machine to the container.
`RUN pip3 install -r requirements.txt`: This installs the dependencies listed in requirements.txt.
`COPY . .`: This copies the current directory (which contains the predict.py file and the model.txt file) to the container.
`CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "80"]`: This specifies the command that should be run when the container starts. In this case, it starts the Uvicorn server that serves the FastAPI app on port 80.

To build the Docker image, run the following command in the same directory as the Dockerfile:
`docker build -t prediction-image .`
This command builds an image with the tag `prediction-image`.

To run the container, use the following command:
`docker run -p 8080:80 prediction-image`
This command starts a container based on the `prediction-image` image and maps port 80 in the container to port 8080 on the host machine.

Access the app by visiting `http://localhost:8080/docs` in your web browser.

Test the predict endpoint with:
`curl -X POST "http://localhost:8080/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"feature1": 0.5, "feature2": 0.7, "feature3": 0.2}'`


# Scaling the app
Here are a few options to scale a FastAPI app, depending on your requirements and resources:

Use a reverse proxy/load balancer: You can use a reverse proxy such as Nginx or a load balancer such as HAProxy to distribute incoming requests across multiple instances of your FastAPI app running on different servers. This can help distribute the load and improve the availability of your app.

Use a container orchestrator: You can use a container orchestrator such as Kubernetes or Docker Swarm to manage and scale your app. With a container orchestrator, you can define the number of replicas or instances of your app that should be running and the orchestrator will handle the deployment, scaling, and load balancing of your app.

Use a cloud provider: Many cloud providers such as AWS, GCP, and Azure offer services for running and scaling containerized applications, such as ECS, EKS, GKE, and AKS. These services provide a managed environment for running your app and can scale automatically based on demand.

Optimize the app: You can optimize your FastAPI app to handle more requests with the same resources. Some techniques include using asynchronous code, caching responses, and optimizing database queries.

Keep in mind that scaling an app can be complex and requires careful planning and monitoring. You should also consider the cost, complexity, and performance tradeoffs of different scaling strategies.
