# Use the official lightweight Python image.
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the web service on container startup. Here we use Streamlit's entrypoint.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
