from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import joblib
import numpy as np
from datetime import datetime
import pickle
import gzip

from utils.preprocessing import preprocess_single_image, extract_single_image_features, validate_pelvic_xray

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'jfif'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables for loaded model
model = None
label_encoder = None
model_accuracy = 0.0

def load_model():
    """Load the trained model"""
    global model, label_encoder, model_accuracy
    try:
        with gzip.open('compressed_file.pkl.gz', 'rb') as f:
            model_data = pickle.load(f)
            
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        model_accuracy = model_data.get('accuracy', 0.0) * 100
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model is not None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction with strict pelvic X-ray validation"""
    if 'file' not in request.files:
        flash('No file uploaded. Please select a pelvic X-ray image to analyze.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected. Please choose a pelvic X-ray image.', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Strict validation for pelvic X-rays
            is_valid_pelvic_xray, validation_message = validate_pelvic_xray(filepath)
            
            if not is_valid_pelvic_xray:
                os.remove(filepath)
                flash(validation_message, 'error')
                return redirect(url_for('index'))
            
            # Check if model is loaded
            if model is None or label_encoder is None:
                if not load_model():
                    flash('Model not available. Please train the model first.', 'error')
                    os.remove(filepath)
                    return redirect(url_for('index'))
            
            # Preprocess image
            img = preprocess_single_image(filepath)
            if img is None:
                os.remove(filepath)
                flash('Could not process image. Please upload a valid pelvic X-ray image.', 'error')
                return redirect(url_for('index'))
            
            # Extract features and predict
            features = extract_single_image_features(img)
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)
            
            gender = label_encoder.inverse_transform(prediction)[0]
            confidence = float(np.max(probabilities))
            
            prediction_data = {
                'gender': gender,
                'confidence': confidence,
                'confidence_percentage': f"{confidence * 100:.2f}%"
            }
            
            
            return render_template('results.html',
                                 prediction=prediction_data,
                                 image_filename=unique_filename,
                                 model_loaded=True,
                                 model_accuracy=f"{model_accuracy:.2f}")
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Prediction failed: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload PNG, JPG, or JPEG pelvic X-ray images only.', 'error')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/confusion-matrix')
def confusion_matrix():
    """Serve confusion matrix image from root directory"""
    try:
        # Check if file exists in current directory
        if not os.path.isfile('confusion_matrix.png'):
            return render_template('error.html',
                                 error_title="Confusion Matrix Not Available",
                                 error_message="Please train the model first to generate the performance metrics.",
                                 model_loaded=model is not None)
        
        # Serve the file from current working directory
        return send_from_directory(os.getcwd(), 'confusion_matrix.png')
        
    except Exception as e:
        print(f"Error in confusion matrix route: {e}")
        return render_template('error.html',
                             error_title="Error Loading Performance Metrics",
                             error_message="There was an error loading the model performance data.",
                             model_loaded=model is not None)

@app.route('/model-status')
def model_status():
    """Check if model is loaded"""
    if model is not None and label_encoder is None:
        load_model()
        
    if model is not None and label_encoder is not None:
        return {
            'status': 'loaded',
            'classes': label_encoder.classes_.tolist(),
            'accuracy': f"{model_accuracy:.2f}%"
        }
    else:
        return {'status': 'not_loaded'}

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return render_template('error.html',
                         error_title="File Too Large",
                         error_message="The pelvic X-ray image you uploaded is too large.",
                         error_details="Maximum file size is 16MB. Please upload a smaller image.",
                         model_loaded=model is not None), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('error.html',
                         error_title="Page Not Found",
                         error_message="The page you're looking for doesn't exist.",
                         model_loaded=model is not None), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('error.html',
                         error_title="Internal Server Error",
                         error_message="Something went wrong on our end. Please try again.",
                         model_loaded=model is not None), 500

# Load model when app starts
@app.before_request
def initialize():
    load_model()

if __name__ == '__main__':
    print("Starting Pelvic X-ray Gender Classification Server...")
    print("Loading model...")
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)