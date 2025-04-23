name: Deploy Lovince AI

on:
  push:
    branches:
      - main

jobs:
  build-backend:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
    - name: Run tests
      run: |
        cd backend
        python -m unittest discover

  build-flutter:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Flutter
      uses: subosito/flutter-action@v2
      with:
        flutter-version: '3.16.0'
    - name: Install dependencies
      run: |
        cd flutter_app
        flutter pub get
    - name: Build APK
      run: |
        cd flutter_app
        flutter build apk --release
    - name: Upload APK
      uses: actions/upload-artifact@v3
      with:
        name: app-release.apk
        path: flutter_app/build/app/outputs/flutter-apk/app-release.apk

  deploy:
    runs-on: ubuntu-latest
    needs: [build-backend, build-flutter]
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to server
      env:
        DEPLOY_SERVER: ${{ secrets.DEPLOY_SERVER }}
        DEPLOY_USER: ${{ secrets.DEPLOY_USER }}
        DEPLOY_PASSWORD: ${{ secrets.DEPLOY_PASSWORD }}
      run: |
        sshpass -p $DEPLOY_PASSWORD ssh $DEPLOY_USER@$DEPLOY_SERVER << 'EOF'
          mkdir -p ~/lovince_ai/backend
          mkdir -p ~/lovince_ai/flutter_app
        EOF
        sshpass -p $DEPLOY_PASSWORD scp -r backend/* $DEPLOY_USER@$DEPLOY_SERVER:~/lovince_ai/backend
        sshpass -p $DEPLOY_PASSWORD ssh $DEPLOY_USER@$DEPLOY_SERVER << 'EOF'
          cd ~/lovince_ai/backend
          pip install -r requirements.txt
          nohup python grok_lovince_unified_core_with_ml.py &
        EOF