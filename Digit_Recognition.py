import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tkinter import Tk, Canvas, Button, Label, YES, BOTH, LEFT


# Loading MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

print("Training images shape:", train_images.shape)
print("Validation images shape:", val_images.shape)
print("Test images shape:", test_images.shape)

# Preprocessing
train_images = train_images.reshape((48000, 28 * 28)).astype('float32') / 255
val_images = val_images.reshape((12000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# One-hot encoding of the labels
def one_hot_encode(labels, dimension=10):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

train_labels = one_hot_encode(train_labels)
val_labels = one_hot_encode(val_labels)
test_labels = one_hot_encode(test_labels)

# Neural network layers and activations
class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.weights) + self.biases
        return self.Z

    def backward(self, dZ, learning_rate):
        dW = np.dot(self.X.T, dZ) / self.X.shape[0]
        db = np.mean(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        return dX
    
class ReLU:
  def __init__(self, leaky = False, alpha = 0.1):
    self.alpha = alpha
    self.leaky = leaky

  def activation_function(self, X):
    if self.leaky:
      return np.maximum(0, self.alpha * X)
    else:
      return np.maximum(0,X)

  def activation_derivative(self, X):
      return X > 0    
    
class Sigmoid:

  def __init__(self):
    pass

  def activation_function(self, X):
      #X = X - np.max(X, axis=1, keepdims=True)
      return 1 / (1 + np.exp(- X ))

  def activation_derivative(self, X):
      return self.activation_function(X) * (1 - self.activation_function(X))
  
class Softmax:
    def activation_function(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def activation_derivative(self, X):
        # This is handled indirectly in the backward pass of the network
        pass

class NeuralNetwork:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations

    def forward(self, X):
        A = X
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            Z = layer.forward(A)
            A = activation.activation_function(Z)
        return A

    def backward(self, X, Y, learning_rate):
        output = self.forward(X)
        dA = output - Y
        for i in reversed(range(len(self.layers))):
            activation = self.activations[i]
            if isinstance(activation, Softmax):
                dZ = dA  # Softmax and cross-entropy derivative
            else:
                dZ = dA * activation.activation_derivative(self.layers[i].Z)
            dA = self.layers[i].backward(dZ, learning_rate)

    def compute_loss(self, X, Y):
        output = self.forward(X)
        return -np.mean(np.sum(Y * np.log(output + 1e-8), axis=1))

'''
layer1 = Layer(784, 128)
layer2 = Layer(128, 10)
relu = ReLU()
softmax = Softmax()
nn = NeuralNetwork([layer1, layer2], [relu, softmax])
'''

#implementing fuctions for deriving user-defined parameters

#deriving activations - handled errors

def get_activation():
    while True:
        activation = input("Choose an activation function (ReLU / Leaky ReLU / Sigmoid): ").lower()
        if activation == "relu":
            return ReLU()
        elif activation == "leaky relu":
            while True:
                alpha = float(input("Enter alpha for Leaky ReLU: "))
                if (isinstance(int(alpha), int)):
                    return ReLU(leaky=True, alpha=alpha)
                
                print("Invalid parameter. Please provide an integer.")
                    
        elif activation == "sigmoid":
            return Sigmoid()
        
        else:
            print("Invalid activation function. Please choose from ReLU, Leaky ReLU, or Sigmoid.")
            
#deriving number of nodes - handled errors

def Get_Parameters():
    while True:
        try:
            num_layers = int(input("Enter the number of layers: "))
            if num_layers < 1:
                print("Please specify at least one hidden layer.")
                continue
            layer_sizes = [784]
            activations = []

            for i in range(num_layers):
                while True:
                    try:
                        size = int(input(f"Enter the number of nodes for layer {i + 1}: "))
                        if size < 1:
                            print("Number of nodes must be a positive integer.")
                            continue
                        layer_sizes.append(size)
                        activation = get_activation()
                        activations.append(activation)
                        break
                    except ValueError:
                        print("Please enter a valid number of nodes.")
                if(i == num_layers - 1):
                    activations.append(Softmax())
                    layer_sizes.append(10)
            print(layer_sizes)            
            return layer_sizes, activations
        except ValueError:
            print("Please enter a valid number of layers.")


layer_sizes, activations = Get_Parameters()
layer_objects = []
for i in range(len(layer_sizes) - 1):
    layer_objects.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
    #print(layer_objects[i].input_size,layer_objects[i].output_size)

#print(activations)
nn = NeuralNetwork(layer_objects, activations)



# Training
epochs = 10
batch_size = 128
learning_rate = 0.1

n_batches = len(train_images) // batch_size

training_loss = []
val_loss = []
for epoch in range(epochs):
    epoch_losses = []
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_images = train_images[batch_start:batch_end]
        batch_labels = train_labels[batch_start:batch_end]

        loss = nn.compute_loss(batch_images, batch_labels)
        epoch_losses.append(loss)
        nn.backward(batch_images, batch_labels, learning_rate)

    average_loss = np.mean(epoch_losses)
    training_loss.append(average_loss)  # Store the loss for this epoch
    val_loss.append(nn.compute_loss(val_images, val_labels))

    print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}')


#If you want to see the Trainin and Validation Losses after each epoch, run this code

"""plt.plot(range(1, epochs + 1), training_loss, label='Training Loss')
plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validaton Losses over Epochs')
plt.legend()
plt.grid(True)
plt.show()
"""



# Evaluation function
def evaluate_model(network, images, labels):
    predictions = network.forward(images)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(labels, axis=1)
    accuracy = np.mean(predicted_labels == actual_labels)
    return accuracy


# Evaluate the model
accuracy = evaluate_model(nn, test_images, test_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


#Testing a build-in model and comparing

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])



# Compiling the model
model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=2)

# Evaluating the Sequential model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Keras Model Test Accuracy: {test_accuracy * 100:.2f}%')

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)


    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array.reshape(1, 784)
    img_array = img_array.astype('float32') / 255
    return img_array

def predict_with_custom_model(model, image):
    prediction = model.forward(image)
    return np.argmax(prediction, axis=1)
def predict_with_keras_model(model, image):
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)

def save_image():
    canvas_img.save("canvas.png")
    preprocessed_image = load_and_preprocess_image("canvas.png")
    custom_model_prediction = predict_with_custom_model(nn, preprocessed_image)
    keras_model_prediction = predict_with_keras_model(model, preprocessed_image)
    
    result_text = f'Custom Model Prediction: {custom_model_prediction[0]}, Keras Model Prediction: {keras_model_prediction[0]}'
    result_label.config(text=result_text)
    print(custom_model_prediction[0])
    print(keras_model_prediction[0])

def clear_canvas():
    canvas.delete("all")
    global canvas_img, draw
    canvas_img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas_img)
    canvas.update()  
    result_label.config(text="")

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=20)

root = Tk()
root.title("Digit Recognition")
canvas_width = 280
canvas_height = 280
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack(expand=YES, fill=BOTH)
canvas_img = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas_img)
canvas.bind("<B1-Motion>", paint)

btn_clear = Button(text="Clear", command=clear_canvas)
btn_clear.pack(side=LEFT)

btn_save = Button(text="Recognize", command=save_image)
btn_save.pack(side=LEFT)

result_label = Label(root, text="")
result_label.pack(side=LEFT)



root.mainloop()
