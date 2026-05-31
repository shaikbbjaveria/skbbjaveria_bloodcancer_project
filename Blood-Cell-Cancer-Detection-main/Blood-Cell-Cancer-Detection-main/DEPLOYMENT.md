# Render Deployment Configuration

## Services Needed

### Web Service
- **Name**: blood-cancer-detection
- **Environment**: Python 3.14.3 (Render default)
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `python start.py`
- **Health Check Path**: `/`

### PostgreSQL Database (Optional)
- **Name**: blood-cancer-db
- **Version**: PostgreSQL 14

## Environment Variables

Set these in your Render dashboard:

```
SECRET_KEY=your-secure-secret-key-here
FLASK_ENV=production
DATABASE_URL=postgresql://user:pass@host:port/dbname
PYTHONUNBUFFERED=1
TF_CPP_MIN_LOG_LEVEL=2
```

## Deployment Steps

1. Connect GitHub repository to Render
2. Create Web Service with Python 3.14.3 environment
3. Set environment variables
4. Deploy - Render will automatically detect the Procfile
5. Test the deployed application

## Important Notes

- Uses Python 3.14.3 (Render's default)
- TensorFlow 2.15.x with latest compatibility
- Pillow 10.1.x (latest stable version)
- Includes fallback handling for TensorFlow import issues
- Uses custom start.py script for better error handling
- All dependencies use minimum version requirements for flexibility

## Troubleshooting

If TensorFlow fails to install:
1. Check the build logs
2. The start.py script includes fallback installation
3. TensorFlow 2.15.x is used for Python 3.14 compatibility

If Pillow fails to install:
1. Python 3.14.3 is used with latest Pillow version
2. Pillow 10.1.x supports Python 3.14
3. Version ranges allow for compatible updates

## Package Versions (Python 3.14 Compatible)
- Python: 3.14.3 (Render default)
- Flask: 2.3.x
- TensorFlow: 2.15.x
- Pillow: 10.1.x
- NumPy: 1.24.x
- All other packages: Latest compatible versions
