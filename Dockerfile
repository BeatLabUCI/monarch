# Use a slim Python image
FROM python:3.11-slim


WORKDIR /app
COPY . /app

# Install system dependencies (optional)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Voilà will run on (Render sets PORT env var)
EXPOSE 8000

# Run Voilà with Render's dynamic port
# Bind to 0.0.0.0 so Render can detect the open port
CMD ["sh", "-c", "voila monarch-docs/docs/monarch_starter_interactive.ipynb --port=${PORT:-10000} --no-browser --ServerApp.ip=0.0.0.0 --template=lab"]