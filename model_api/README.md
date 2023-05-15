# Running the app
To run the application, first train the model by running `python train_model.py`. This will save the trained model to a file named `model.txt`.

Then, start the FastAPI app by running `uvicorn predict:app --reload`. 
This will start the app and listen for incoming requests at `http://localhost:8000`. 
You can test the API by sending a POST request to `http://localhost:8000/predict` with a JSON payload containing values for the `feature1`, `feature2`, and `feature3` fields.

The following will send a GET request to your endpoint and return the response.
`curl http://localhost:8000/predict`

Send a POST request instead and include the required data in the request body.
For example, if your endpoint requires a JSON payload containing the data to be predicted, you can use the following command to send a POST request with the required data:
`curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"feature1": 0.5, "feature2": 0.7, "feature3": 0.2}'`

Or using the `payload.json` file that is available:
`curl -X POST -H "Content-Type: application/json" -d @payload.json http://localhost:8000/predict`

This should return a prediction that resembles:
`{"prediction":0.9130092931232355}`

# Scaling the app
There are different ways to scale a FastAPI app, depending on your requirements and resources. Here are a few options:

Use a reverse proxy/load balancer: You can use a reverse proxy such as Nginx or a load balancer such as HAProxy to distribute incoming requests across multiple instances of your FastAPI app running on different servers. This can help distribute the load and improve the availability of your app.

Use a container orchestrator: You can use a container orchestrator such as Kubernetes or Docker Swarm to manage and scale your app. With a container orchestrator, you can define the number of replicas or instances of your app that should be running and the orchestrator will handle the deployment, scaling, and load balancing of your app.

Use a cloud provider: Many cloud providers such as AWS, GCP, and Azure offer services for running and scaling containerized applications, such as ECS, EKS, GKE, and AKS. These services provide a managed environment for running your app and can scale automatically based on demand.

Optimize the app: You can optimize your FastAPI app to handle more requests with the same resources. Some techniques include using asynchronous code, caching responses, and optimizing database queries.

Keep in mind that scaling an app can be complex and requires careful planning and monitoring. You should also consider the cost, complexity, and performance tradeoffs of different scaling strategies.
