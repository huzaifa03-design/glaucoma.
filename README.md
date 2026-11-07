# 👁️ AI-Based Glaucoma Detection

An AI-powered web application that uses Deep Learning and Convolutional Neural Networks (CNNs) to analyze retinal fundus images and detect signs of Glaucoma.

The project uses a **ResNet18-based CNN model** to classify retinal images and provide prediction confidence scores through an interactive Streamlit interface.

## 🌐 Live Demo

**Try the application:** https://glaucoma-detect.streamlit.app

## 🚀 Features

- **Retinal Image Upload:** Upload retinal fundus images for analysis.
- **AI-Based Detection:** Uses a ResNet18 deep learning model for glaucoma classification.
- **Image Preprocessing:** Processes input images into the required format for model inference.
- **Prediction Results:** Displays the predicted class based on the model's output.
- **Confidence Scores:** Provides Softmax-based confidence scores for predictions.
- **Interactive Dashboard:** Simple and user-friendly interface built with Streamlit.
- **Model Evaluation:** Includes performance evaluation using a confusion matrix and classification report.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| ResNet18 | CNN architecture for image classification |
| OpenCV | Image processing |
| Streamlit | Web application interface |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |

## 🧠 Model Architecture

The project uses **ResNet18 (Residual Neural Network)**, a convolutional neural network architecture designed for image classification.

The model analyzes retinal fundus images to identify patterns associated with glaucoma.

**Workflow:**

1. Input retinal fundus image.
2. Image preprocessing and resizing.
3. Feature extraction using ResNet18.
4. Classification through the output layer.
5. Softmax-based confidence score generation.
6. Display prediction results on the Streamlit dashboard.

## ⚙️ Project Workflow

```text
Retinal Fundus Image
         |
         v
   Image Upload
         |
         v
 Image Preprocessing
         |
         v
   ResNet18 CNN
         |
         v
 Feature Extraction
         |
         v
  Classification
         |
         v
 Prediction + Confidence
         |
         v
  Streamlit Dashboard
```

## 📂 Project Structure

```text
glaucoma/
│
├── app.py                  # Streamlit application
├── model/                  # Trained model files
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── ...
```

*Note: Update the project structure above to match your actual repository files.*

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/huzaifa03-design/glaucoma.git
```

### 2. Navigate to the project directory

```bash
cd glaucoma
```

### 3. Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📊 Model Evaluation

The model's performance can be evaluated using:

- **Confusion Matrix:** Visualizes correct and incorrect classifications.
- **Classification Report:** Provides precision, recall, and F1-score.
- **Softmax Confidence:** Displays the model's confidence for each prediction.

## 🔮 Future Enhancements

- Improve model performance through additional training and fine-tuning.
- Integrate Grad-CAM for visual explanations of model predictions.
- Add retinal image quality assessment.
- Explore larger CNN architectures and transfer learning techniques.
- Develop a patient history and prediction tracking dashboard.
- Deploy the application with scalable cloud infrastructure.

## ⚠️ Disclaimer

This project is developed for educational and research purposes. It is not a substitute for professional ophthalmological examination or medical diagnosis. Predictions should be reviewed by qualified healthcare professionals.

