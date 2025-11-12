# Use offical python image
FROM python:3.10-slim

#Set working directory
WORKDIR /app

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt
# Copy only requirements first (for efficient caching)
COPY requirements.txt .

# Install system dependencies (needed for numpy, scikit-learn)
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

#Copy FILES INTO CONTAINER
COPY . .
# Expose post
EXPOSE 8200

# Command to run the server

CMD ["uvicorn", "iris_log:app", "--host", "0.0.0.0", "--port", "8200"]


