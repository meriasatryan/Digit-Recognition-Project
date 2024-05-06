# **MNIST Digit Recognition README**
## **Overview**

This repository contains code for a simple digit recognition application using the MNIST dataset. It includes both a custom neural network built from scratch and a built-in model using TensorFlow's Keras API.

## **Features**

· Custom-built neural network using NumPy.

· Neural network implementation with TensorFlow Keras.

· Image preprocessing and digit recognition from user input.

· GUI for drawing digits and recognizing them in real-time.

## **Installation**

Ensure you have Python 3.6+ installed on your system. Then, install the required dependencies:

To install the necessary Python packages using `pip`, open your terminal or command prompt and run the following command:
```bash
pip install numpy opencv-python matplotlib pillow tensorflow
```

## **Usage**

To run the digit recognition GUI, execute the following command:

```bash
python digit_recognition.py
```

Draw a digit on the canvas and click the "Recognize" button to see the prediction from both the custom and Keras models. Use the "Clear" button to reset the canvas.

## **Training the Model**

The script will automatically train both the custom and the Keras model when you run it. It trains on the MNIST dataset, which is automatically downloaded the first time you run the script.

## **Predictions**

Once the models are trained, you can test them by drawing digits on the provided canvas. The script preprocesses the canvas image and feeds it to both models to compare their predictions.

## **Code Structure**

**Neural Network Definition**: The custom model layers are defined using NumPy, with a  **Layer**  class for linear layers,  **ReLU**  for activation, and  **Softmax**  for the output layer.

**Keras Model**: A sequential model is defined using TensorFlow's high-level API for ease of use and performance.

**GUI**: A simple GUI using Tkinter for drawing and recognizing digits.

**Image Preprocessing**: Functions to convert canvas drawings into suitable model input.

## **Files**

**digit_recognition.py**: Main Python script with GUI and model training logic.

**README.md**: This file, describing the project and how to use it.

## **Dependencies**

· NumPy

· OpenCV

· Matplotlib

· PIL

· TensorFlow (includes Keras)

· Tkinter (standard library)

## **Evaluation**

After training, the script will output the accuracy of the models on the MNIST test dataset. This provides a straightforward comparison between the custom and Keras models.

Here are the results from the evaluation:

· **Custom Model Accuracy:** 96.50%

· **Keras Model Accuracy:** 96.85%

## **Authors**
Ani Aloyan
Meri Asatryan
Seda Bayadyan

