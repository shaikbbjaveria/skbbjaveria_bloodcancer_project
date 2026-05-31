# Blood Cell Cancer Detection

A Flask-based web application for detecting blood cancer using machine learning. The app analyzes blood cell images and predicts whether cancer is present, along with the stage and severity.

## Features

- **Image Analysis**: Upload blood cell images for cancer detection
- **Multi-stage Detection**: Identifies different stages of blood cancer (Early Pre-B, Pre-B, Pro-B)
- **User Management**: Role-based access for patients, doctors, and admins
- **Real-time Predictions**: Uses TensorFlow Lite model for fast inference
- **Secure Authentication**: Login system with role-based permissions
- **Database Integration**: Stores prediction history and user data

## Technology Stack

- **Backend**: Flask, SQLAlchemy, Flask-Login
- **Machine Learning**: TensorFlow, MobileNetV2
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: PostgreSQL (production), SQLite (development)
- **Deployment**: Render

## Model Information

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Classes**: 4 (Benign, Malignant early Pre-B, Malignant Pre-B, Malignant Pro-B)
- **Accuracy**: 80% on test dataset
- **Input Size**: 224x224 RGB images
- **Model Size**: 2.68 MB (TensorFlow Lite)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/saiprudhvi01/Blood-Cell-Cancer.git
cd Blood-Cell-Cancer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open `http://localhost:8080` in your browser.

### Production Deployment (Render)

The app is configured for easy deployment on Render:

1. Push to GitHub
2. Connect repository to Render
3. Render will automatically build and deploy

## Environment Variables

For production deployment, set these environment variables:

- `SECRET_KEY`: Flask secret key
- `DATABASE_URL`: PostgreSQL connection string
- `FLASK_ENV`: Set to 'production'

## Usage

1. **Register**: Create an account as a patient, doctor, or admin
2. **Login**: Access the dashboard based on your role
3. **Upload Image**: Patients can upload blood cell images for analysis
4. **View Results**: Get instant predictions with confidence scores
5. **History**: Track previous predictions and results

## API Endpoints

- `POST /predict`: Analyze blood cell image (requires authentication)
- `POST /test-predict`: Test endpoint without authentication
- `GET /`: Home page
- `GET /patient/dashboard`: Patient dashboard
- `GET /doctor/dashboard`: Doctor dashboard

## Dataset

The model was trained on blood cell images with the following classes:
- Benign (Healthy)
- Malignant early Pre-B (Early stage)
- Malignant Pre-B (Developing stage)  
- Malignant Pro-B (Advanced stage)

## Performance Metrics

- **Overall Accuracy**: 80%
- **Benign Detection**: 100%
- **Early Pre-B Detection**: 100%
- **Pro-B Detection**: 100%
- **Pre-B Detection**: 60%

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on GitHub.
