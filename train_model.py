import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # To handle class imbalance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import gzip

from utils.preprocessing import load_images_from_directory, extract_features

def train_model():
    print("🚀 Starting enhanced model training...")
    dataset_path = 'new_dataset' 
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset folder '{dataset_path}' not found!")
        return None, None, 0
    
    X, y = load_images_from_directory(dataset_path)
    print(f"📊 Loaded {len(X)} total images (F: 216, M: 343)")
    
    print("🔍 Extracting features...")
    X_features = extract_features(X)
    
    # 1. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 2. Split data first (to keep test set pure)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 3. Address Imbalance with SMOTE (Only on Training Data)
    print("⚖️ Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"📈 New training distribution: {np.unique(y_train_res, return_counts=True)}")

    # 4. Pipeline: Standardize -> PCA -> SVM
    # SVMs are generally more robust for high-dimensional feature vectors than Random Forests
    print("🤖 Building Pipeline (StandardScaler + PCA + SVM)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # SVM requires scaled data
        ('pca', PCA(n_components=0.95)), # Keep 95% of variance
        ('clf', SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42))
    ])
    
    pipeline.fit(X_train_res, y_train_res)
    
    print("📋 Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Optimized SVM')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    model_data = {
        'model': pipeline,
        'label_encoder': le,
        'accuracy': accuracy
    }
    
    with gzip.open('compressed_file.pkl.gz', 'wb') as f:
        pickle.dump(model_data, f)
    print("💾 Model saved as 'compressed_file.pkl.gz'")
    
    return pipeline, le, accuracy

if __name__ == '__main__':
    train_model()