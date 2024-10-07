# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt to install dependencies
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run the initialization script for LanceDB and start the Streamlit app
CMD ["sh", "-c", "python init_lancedb.py && streamlit run app.py"]
