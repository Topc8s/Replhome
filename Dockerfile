# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    nodejs \
    npm \
    default-jdk \
    g++ \
    golang \
    ruby \
    php \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p projects static templates logs temp

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "autodev_ide.py"]

---

# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - DATABASE_URL=postgresql://autodev:password@db:5432/autodev
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./projects:/app/projects
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=autodev
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=autodev
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:

---

# setup.sh
#!/bin/bash

echo "🚀 AutoDev IDE Setup Script"
echo "=========================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p projects static templates logs temp

# Setup database
echo "🗄️ Initializing database..."
python -c "from autodev_ide import Base, engine; Base.metadata.create_all(bind=engine)"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOL
OPENAI_API_KEY=your-openai-api-key-here
GITHUB_TOKEN=your-github-token-here
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
DATABASE_URL=sqlite:///./autodev.db
REDIS_URL=redis://localhost:6379
EOL
    echo "⚠️  Please edit .env file and add your API keys"
fi

# Install additional language runtimes (optional)
echo "🔧 Checking language runtimes..."
command -v node >/dev/null 2>&1 || echo "⚠️  Node.js not found. Install for JavaScript support."
command -v go >/dev/null 2>&1 || echo "⚠️  Go not found. Install for Go support."
command -v rustc >/dev/null 2>&1 || echo "⚠️  Rust not found. Install for Rust support."
command -v java >/dev/null 2>&1 || echo "⚠️  Java not found. Install for Java support."

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Edit .env file with your API keys"
echo "  3. Run: python autodev_ide.py"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo ""

---

# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
