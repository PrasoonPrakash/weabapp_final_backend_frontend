# aanchal_ai_webapp

1. Copy all the codes to the server. Let <app_path> be the application path. `cd <app_path>`
2. Build the Docker image. `docker build -t aanchal-webapp .`
3. Run the service using following command.
```
docker run -d -p <host_port>:8080 -v <app_path>:/app aanchal-webapp
```
