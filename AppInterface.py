import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PneumoniaDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the window properties
        self.setWindowTitle('Pneumonia Detection App')
        self.setGeometry(100, 100, 600, 400)

        # Initialize label for image preview
        self.image_label = QLabel('No image uploaded')
        self.image_label.setAlignment(Qt.AlignCenter)

        # Initialize label for displaying prediction result
        self.result_label = QLabel('')
        self.result_label.setAlignment(Qt.AlignCenter)

        # Initialize upload button
        self.upload_button = QPushButton('Upload Chest X-ray')
        self.upload_button.clicked.connect(self.upload_image)

        # Load the pre-trained VGG16-based pneumonia detection model
        self.model = load_model('pneumonia_vgg16_model.h5')

        # Set up vertical layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def upload_image(self):
        """
        Opens a file dialog for the user to select an image.
        The selected image is displayed in the GUI and passed to the model for prediction.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Chest X-ray Image', '',
                                                   'Image Files (*.png *.jpg *.jpeg)', options=options)
        if file_path:
            # Display selected image
            pixmap = QPixmap(file_path).scaled(300, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            # Predict pneumonia status from image
            prediction = self.predict_pneumonia(file_path)

            # Update result label with prediction
            self.result_label.setText(prediction)

    def predict_pneumonia(self, image_path):
        """
        Loads the image, preprocesses it, and returns the model's prediction.
        """
        try:
            # Load image with target size as required by VGG16
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)

            # Normalize pixel values and expand dimensions for model input
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction using loaded model
            prediction = self.model.predict(img_array)[0][0]

            # Interpret prediction
            if prediction >= 0.5:
                return "Result: Pneumonia Detected"
            else:
                return "Result: Normal (No Pneumonia)"
        except Exception as e:
            return f"Error processing image: {str(e)}"

if __name__ == '__main__':
    # Create application and run main window
    app = QApplication(sys.argv)
    window = PneumoniaDetectionApp()
    window.show()
    sys.exit(app.exec_())
