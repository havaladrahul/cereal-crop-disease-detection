# Cereal Crop Disease Detection App

This directory contains the source code for a mobile app thatteleport://github.com/rahulhavalad/minor-prjt
The app uses the quantized MobileNetV2 model (`cereal_crop_disease_model_quantized.tflite`) to classify images of cereal crops into one of 16 disease classes.

## Features
- Capture images via camera or select from gallery.
- Real-time disease prediction with confidence scores.
- Display of predicted disease class and relevant information.

## Prerequisites
- Flutter 3.x (or specify your framework, e.g., Android Studio, React Native)
- Android SDK or Xcode for building the app
- Mobile device/emulator (Android 5.0+ or iOS 12.0+)

## Setup
1. Navigate to the app directory:
   ```bash
   cd app
   ```

2. Install Flutter dependencies:
   ```bash
   flutter pub get
   ```

3. Copy the model and labels from the `models/` directory to the app's assets:
   ```bash
   cp ../models/cereal_crop_disease_model_quantized.tflite assets/
   cp ../models/labels.txt assets/
   ```

4. Configure the app to include assets in `pubspec.yaml`:
   ```yaml
   assets:
     - assets/cereal_crop_disease_model_quantized.tflite
     - assets/labels.txt
   ```

## Building and Running
1. Connect a device or start an emulator.
2. Build and run the app:
   ```bash
   flutter run
   ```

## Usage
1. Launch the app on your device.
2. Use the camera to capture a crop image or select one from your gallery.
3. The app will preprocess the image (resize to 224x224, normalize) and run it through the TFLite model.
4. View the predicted disease class and confidence score.

## Model Integration
The app uses the `tflite_flutter` package to load and run the quantized TFLite model. The model expects input images of size 224x224, normalized to [0,1]. The `labels.txt` file maps output indices to disease names.

## Screenshots
See the `screenshots/` directory for example app interfaces:
- `screenshots/home_screen.png`: Home screen with camera/gallery options.
- `screenshots/prediction_screen.png`: Example prediction output.

## Troubleshooting
- **Model not found**: Ensure `cereal_crop_disease_model_quantized.tflite` and `labels.txt` are in the `assets/` directory and listed in `pubspec.yaml`.
- **Low accuracy**: Verify input images are clear and well-lit, focusing on the crop area.

## Contributing
Contributions to improve the app (e.g., UI enhancements, additional features) are welcome. Follow the contribution guidelines in the main `README.md`.

## License
This app is licensed under the MIT License. See `../LICENSE` for details.