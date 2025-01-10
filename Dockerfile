FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential libssl-dev libffi-dev python3-dev

# Install Rust and Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port 5000 for Flask app
EXPOSE 5000

# Command to run the app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]