FROM python:3.9

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    wget -O config.guess 'http://savannah.gnu.org/cgi-bin/viewcvs/*checkout*/config/config/config.guess' && \
    wget -O config.sub 'http://savannah.gnu.org/cgi-bin/viewcvs/*checkout*/config/config/config.sub' && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 80
EXPOSE 80

# Command to run the application
CMD ["python", "main.py"]