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