🧠 Handwritten Digit Recognition (MNIST)

This project implements Handwritten Digit Recognition using the MNIST dataset
.
It includes training both a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), comparing their performance, visualizing results, and deploying with Streamlit.

⚡ Features

Load and preprocess MNIST dataset

Train and evaluate MLP and CNN models

Compare performance (MLP vs CNN)

Plot accuracy & loss curves

Visualize predictions with correct/incorrect coloring

Generate confusion matrix + classification report

Save the CNN model (cnn_mnist.h5) for deployment

Deploy an interactive app with Streamlit

🚀 How to Run Training

Clone the repository:

git clone https://github.com/Melissiasamir/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition


Install dependencies:

pip install -r requirements.txt


Run the training script:

python main.py


After training, the model will be saved automatically as:

saved_models/cnn_mnist.h5


📈 Results

MLP Test Accuracy: ~97–98%

CNN Test Accuracy: ~99% ✅

👉 CNN consistently performs better than MLP for handwritten digit recognition.

Example output:

Accuracy & Loss plots during training

Confusion Matrix

Predictions with green (correct) / red (incorrect) labels

🔮 Run the Streamlit App

After training and saving the model, you can launch the interactive app:

streamlit run app.py


Features:

🎨 Draw digits on a canvas

📤 Upload digit images

🔮 Predict digit with probability chart



