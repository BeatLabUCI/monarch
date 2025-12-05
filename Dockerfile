# Use a slim Python image
FROM python:3.11-slim


WORKDIR /app
COPY . /app

# Install system dependencies (optional)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install the monarch package from the current directory
RUN pip install --no-cache-dir .

# Expose the port Voilà will run on (Render sets PORT env var)
EXPOSE $PORT

# Set environment variable to bind to all interfaces
ENV JUPYTER_IP=0.0.0.0

# Run Voilà with Render's dynamic port
# Bind to 0.0.0.0 so Render can detect the open port
CMD ["sh", "-c", "voila monarch-docs/docs/monarch_starter_interactive.ipynb --port=$PORT --no-browser --Voila.ip=0.0.0.0 --template=lab"]
