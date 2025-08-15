# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Make port 8081 available to the world outside this container
EXPOSE 8081

# Define the command to run your app using Gunicorn (a production-ready server)
# It will run the 'app' object from your 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:8081", "app:app"]