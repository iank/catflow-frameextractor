# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container to /app
WORKDIR /app

# Install ffmpeg, libsm6, libxext6
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install torch cpu version specifically
ADD requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Install any needed packages specified in requirements.txt
ADD dist/frameextractor-0.5.1-py3-none-any.whl /app
RUN pip install --no-cache-dir frameextractor-0.5.1-py3-none-any.whl

# We'll add these with docker-compose for now
#ADD frameextractor.ini /app
#ADD lively_station_9801_best.pt /app

# Make port 5053 available to the world outside this container
EXPOSE 5053

# Run app.py when the container launches
CMD ["waitress-serve", "--port=5053", "--call", "frameextractor:create_app"]
