Cereal Crop Disease Detection
This project implements a deep learning model to detect diseases in cereal crops (Corn, Rice, Sorghum, Wheat) using MobileNetV2, trained on a dataset of 16 disease classes. The model is quantized to TFLite for efficient deployment in a mobile app, achieving a test accuracy of 95.55%. The repository includes the training notebook, quantized model, labels, and a mobile app for real-world use.
Features

Model: MobileNetV2-based classifier, fine-tuned for 16 cereal crop disease classes.
Dataset: Cereal Crop Disease Dataset from Kaggle.
App: Mobile app for capturing images and predicting crop diseases using the TFLite model.
Outputs: Quantized TFLite model (cereal_crop_disease_model_quantized.tflite) and labels (labels.txt).

Dataset
The dataset is sourced from Kaggle (rahulhavalad/minor-prjt) and contains images of cereal crops across 16 classes:

Corn: Blight, Common Rust, Gray Leaf Spot, Healthy
Rice: Bacterial Blight, Blast, Brown Spot
Sorghum: Anthracnose Red Rot, Cereal Grain Molds, Head Smut, Loose Smut, Rust
Wheat: Brown Rust, Healthy, Septoria, Yellow Rust

The dataset is split into Train (10,844 images), Validation (1,348 images), and Test (1,371 images) directories.
Installation
Prerequisites

Python 3.11
GPU (recommended for training)
Mobile device/emulator for app (Android/iOS)

Setup

Clone the repository:
git clone https://github.com/<your-username>/cereal-crop-disease-detection.git
cd cereal-crop-disease-detection


Install dependencies:
pip install -r requirements.txt


Download the dataset using KaggleHub:
import kagglehub
path = kagglehub.dataset_download("rahulhavalad/minor-prjt")
print("Path to dataset files:", path)


For app setup, see app/README.md.


Usage
Training the Model

Open notebooks/eis-project.ipynb in Jupyter Notebook or Kaggle.
Update the dataset paths (TRAIN_DIR, VALIDATION_DIR, TEST_DIR) to match the downloaded dataset location.
Run all cells to train the model, evaluate on the test set, and save the quantized TFLite model and labels.

Running Inference
Use the provided scripts/predict.py for single-image predictions:
python scripts/predict.py --image path_to_image.jpg --model models/cereal_crop_disease_model_quantized.tflite --labels models/labels.txt

Using the Mobile App
Refer to app/README.md for instructions on building and running the app.
Model Details

Architecture: MobileNetV2 (pre-trained on ImageNet), with a global average pooling layer, dropout (0.3), and a dense layer for 16 classes.
Training: 40 epochs max, early stopping (patience=5), Adam optimizer (learning rate=0.0005), sparse categorical crossentropy loss.
Test Accuracy: 95.55% (evaluated on test dataset).
Output: Quantized TFLite model (INT8) for efficient deployment on mobile devices.

Visualizations

Training History: Loss and accuracy plots (visualizations/training_history.png).
Confusion Matrix: Per-class performance (visualizations/confusion_matrix.png).


Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Submit a pull request.

Report issues or suggest features via GitHub Issues.
License
This project is licensed under the MIT License. See LICENSE for details.
Contact
For questions or collaboration, open an issue on GitHub or contact [your-email@example.com].
