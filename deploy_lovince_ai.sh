#!/bin/bash

# deploy_lovince_ai.sh: Automate Lovince AI setup and deployment
# Usage: ./deploy_lovince_ai.sh [setup|test|deploy]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Environment variables
PYTHON_VERSION="3.8"
FLUTTER_VERSION="stable"
API_PORT=5000
SERVER_IP="your-server"  # Replace with your server IP
APP_NAME="lovince_ai"

# Logging function
log() {
  echo -e "${GREEN}[INFO] $1${NC}"
}

error() {
  echo -e "${RED}[ERROR] $1${NC}"
  exit 1
}

# Setup environment
setup() {
  log "Setting up Lovince AI environment..."

  # Install Python
  if ! command -v python$PYTHON_VERSION &> /dev/null; then
    log "Installing Python $PYTHON_VERSION..."
    sudo apt-get update
    sudo apt-get install -y python$PYTHON_VERSION python$PYTHON_VERSION-venv
  fi

  # Create virtual environment
  python$PYTHON_VERSION -m venv venv
  source venv/bin/activate

  # Install Python dependencies
  log "Installing Python dependencies..."
  pip install --upgrade pip
  pip install qiskit qiskit-aer numpy matplotlib flask cryptography scipy || error "Python dependencies failed"

  # Install Flutter
  if ! command -v flutter &> /dev/null; then
    log "Installing Flutter..."
    sudo snap install flutter --classic
    flutter channel $FLUTTER_VERSION
    flutter upgrade
  fi

  # Install Flutter dependencies
  cd flutter_app
  flutter pub get || error "Flutter dependencies failed"
  cd ..

  log "Setup complete!"
}

# Test the application
test() {
  log "Testing Lovince AI..."

  source venv/bin/activate

  # Test Python backend
  python -m unittest discover -s tests -p "test_*.py" || error "Python tests failed"

  # Test Flutter app
  cd flutter_app
  flutter test || error "Flutter tests failed"
  cd ..

  log "All tests passed!"
}

# Deploy to server
deploy() {
  log "Deploying Lovince AI..."

  # Start Flask API
  source venv/bin/activate
  nohup python lovince_ai_api.py --port $API_PORT &> lovince_ai.log &
  log "Flask API started on port $API_PORT"

  # Build Flutter APK
  cd flutter_app
  flutter build apk --release
  log "APK built: build/app/outputs/flutter-apk/app-release.apk"

  # TODO: Upload to Play Store (manual step or use fastlane)
  log "Ready for Play Store upload. Follow Google Play Console instructions."
}

# Main logic
case "$1" in
  setup)
    setup
    ;;
  test)
    test
    ;;
  deploy)
    deploy
    ;;
  *)
    error "Usage: $0 [setup|test|deploy]"
    ;;
esac


#!/bin/bash

# deploy_lovince_ai.sh: Automate Lovince AI setup, test, and deployment
# Usage: ./deploy_lovince_ai.sh [setup|test|deploy]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Environment variables
PYTHON_VERSION="3.8"
FLUTTER_VERSION="stable"
API_PORT=5000
SERVER_IP="your-server"  # Replace with your server IP
APP_NAME="lovince_ai"
LOG_FILE="deploy.log"

# Logging function
log() {
  echo -e "${GREEN}[INFO] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
  echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
  exit 1
}

# Check dependencies
check_deps() {
  log "Checking dependencies..."
  for cmd in python$PYTHON_VERSION flutter curl; do
    command -v $cmd &> /dev/null || error "$cmd is required but not installed"
  done
}

# Setup environment
setup() {
  log "Setting up Lovince AI environment..."

  # Install Python
  if ! command -v python$PYTHON_VERSION &> /dev/null; then
    log "Installing Python $PYTHON_VERSION..."
    sudo apt-get update
    sudo apt-get install -y python$PYTHON_VERSION python$PYTHON_VERSION-venv
  fi

  # Create and activate virtual environment
  python$PYTHON_VERSION -m venv venv
  source venv/bin/activate

  # Install Python dependencies
  log "Installing Python dependencies..."
  pip install --upgrade pip
  pip install qiskit qiskit-aer numpy matplotlib flask cryptography scipy || error "Python dependencies failed"

  # Install Flutter
  if ! command -v flutter &> /dev/null; then
    log "Installing Flutter..."
    sudo snap install flutter --classic
    flutter channel $FLUTTER_VERSION
    flutter upgrade
  fi

  # Install Flutter dependencies
  log "Installing Flutter dependencies..."
  cd flutter_app
  flutter pub get || error "Flutter dependencies failed"
  cd ..

  log "Setup complete!"
}

# Test the application
test() {
  log "Testing Lovince AI..."

  source venv/bin/activate

  # Test Python backend
  log "Running Python tests..."
  python -m unittest discover -s tests -p "test_*.py" || error "Python tests failed"

  # Test Flutter app
  log "Running Flutter tests..."
  cd flutter_app
  flutter test || error "Flutter tests failed"
  cd ..

  log "All tests passed!"
}

# Deploy to server
deploy() {
  log "Deploying Lovince AI..."

  # Start Flask API
  source venv/bin/activate
  log "Starting Flask API on port $API_PORT..."
  nohup python lovince_ai_api.py --port $API_PORT &> lovince_ai_api.log &
  sleep 2
  if ! ps aux | grep -q "[p]ython lovince_ai_api.py"; then
    error "Flask API failed to start"
  fi

  # Build Flutter APK
  log "Building Flutter APK..."
  cd flutter_app
  flutter build apk --release || error "Flutter build failed"
  log "APK built: build/app/outputs/flutter-apk/app-release.apk"

  # Verify server connectivity
  log "Verifying API connectivity..."
  curl -s -o /dev/null -w "%{http_code}" http://$SERVER_IP:$API_PORT/run_quantum | grep -q 200 || error "API not reachable"

  log "Deployment complete! Ready for Play Store upload."
}

# Main logic
check_deps
case "$1" in
  setup)
    setup
    ;;
  test)
    test
    ;;
  deploy)
    deploy
    ;;
  *)
    error "Usage: $0 [setup|test|deploy]"
    ;;
esac