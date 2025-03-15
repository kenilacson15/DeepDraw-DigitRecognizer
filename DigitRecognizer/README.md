# Digit Recognizer

Digit Recognizer is an advanced application that leverages convolutional neural networks (CNNs) to accurately recognize handwritten digits. Built upon the MNIST dataset, this project showcases state-of-the-art machine learning techniques through an intuitive graphical user interface.

![Digit Recognizer Screenshot](docs/screenshot.png)

## Features

- **Real-Time Recognition**: Instantly predicts handwritten digits.
- **Interactive Visualization**: Displays confidence scores for all digit classes.
- **Augmentation Preview**: Demonstrates data augmentation effects applied during training.
- **Drawing Playback**: Records and replays the drawing process.
- **User Feedback Collection**: Enables feedback submission to refine model performance.
- **Heatmap Visualization**: Highlights influential regions within the digit image.

## Installation

### Prerequisites

- Python 3.8 or later
- pip package manager

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/digit-recognizer.git
   cd digit-recognizer
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Application:**
   ```bash
   ./run.sh
   ```
   Alternatively, execute:
   ```bash
   python main.py
   ```

## Usage

### Basic Operation

1. Draw a digit (0–9) on the canvas using your mouse.
2. The application will automatically generate a prediction.
3. Review the displayed confidence scores and prediction outcome.
4. Click the "Clear" button to reset the canvas for a new drawing.

### Advanced Functionality

- **Augmentation Preview**: Click "Generate Augmentations" to visualize transformed variations of your input.
- **Drawing Playback**: Use Play/Stop controls to review the drawing process.
- **User Feedback**: Provide feedback by selecting "Yes" or "No" to indicate prediction accuracy.
- **Command-Line Options:**
  ```bash
  # Train a new model
  python main.py --train
  
  # Use a custom model
  python main.py --model-path models/my_custom_model.keras
  
  # Analyze collected feedback
  python main.py --analyze-feedback
  ```

## Project Structure

```
digit_recognizer/
├── main.py                 # Main application entry point
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── run.sh                  # Script to run the application
├── data/                   # Data directories
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed images
│   └── feedback/           # User feedback data
├── models/                 # Trained models directory
│   └── mnist_model.keras   # Default model
├── src/                    # Source code
│   ├── gui/                # Graphical user interface components
│   │   └── digit_recognizer.py
│   ├── model/              # Machine learning model code
│   │   ├── build_model.py  # Model architecture definition
│   │   └── train.py        # Training routines
│   └── utils/              # Utility functions
│       └── image_utils.py  # Image processing utilities
└── tests/                  # Unit tests
    ├── test_image_utils.py
    └── test_model.py
```

## Technical Overview

### Model Architecture

The application employs a convolutional neural network (CNN) with the following structure:
- **Input Layer**: Accepts 28×28 grayscale images.
- **Data Augmentation**: Applies rotations, zoom, and other transformations.
- **Convolutional Layers**: Two layers with max pooling for feature extraction.
- **Regularization**: Dropout layers to mitigate overfitting.
- **Dense Layers**: Fully connected layers for classification.
- **Output Layer**: Produces probabilities for 10 digit classes (0–9).

### Performance Enhancements

- **Optimized Preprocessing**: Streamlined image preprocessing pipeline.
- **Lazy Loading**: Deferred loading of TensorFlow models for efficiency.
- **Optimized Rendering**: Smooth and responsive drawing experience.
- **Memory Management**: Robust handling of large datasets.

## Development

### Running Tests

To execute all unit tests:
```bash
python -m unittest discover tests
```

### Training a Custom Model

To train a new model, run:
```bash
python -m src.model.train
```

### Analyzing User Feedback

To analyze collected user feedback, execute:
```bash
python main.py --analyze-feedback
```

## Contributing

Contributions are welcome. To propose enhancements or report issues:

1. **Fork the repository.**
2. **Create a new branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes:**
   ```bash
   git commit -m 'Add an amazing feature'
   ```
4. **Push your branch:**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Submit a pull request for review.**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The creators of the MNIST dataset.
- The TensorFlow and Keras teams.
- The open-source community for their contributions.

---

Created by [Ken Ira Lacson](https://github.com/KenIraLacson)  
MIT License © 2025 Ken Ira Lacson

