# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Upgrade pip and install all Python dependencies in a single layer
RUN apt-get update && apt-get install -y net-tools && \
    pip install --upgrade pip && \
    pip install deap numpy pygame pymunk debugpy torch torchvision \
    pip install --upgrade pip && pip install -r requirements.txt

ARG FOLDER_NAME=pyAgent_C4
ARG SCRIPT_NAME=train_alphazero.py

ENV FOLDER_NAME=${FOLDER_NAME}
ENV SCRIPT_NAME=${SCRIPT_NAME}

WORKDIR /${FOLDER_NAME}

COPY . /${FOLDER_NAME}

# List the contents of the directory to verify main.py is copied
RUN echo "Contents of /${FOLDER_NAME}:" && ls -la /${FOLDER_NAME}

# Ensure the script exists before changing permissions
RUN if [ -f /${FOLDER_NAME}/${SCRIPT_NAME} ]; then chmod +x /${FOLDER_NAME}/${SCRIPT_NAME}; else echo "Script not found: /${FOLDER_NAME}/${SCRIPT_NAME}"; fi

# Set the DISPLAY environment variable
ENV DISPLAY=host.docker.internal:0.0

# Expose the debug port
EXPOSE 5680

# Start the debugpy server and run the Python script
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5680", "/${FOLDER_NAME}/${SCRIPT_NAME}"]