from flask import Flask, request, render_template, url_for
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the model and prepare the data
model = load_model('mnist_mlp_model.h5')
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_test = x_test.reshape(10000,784).astype('float32') / 255.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Select a random image from the test data and save it to a file
    xhat_idx = np.random.choice(x_test.shape[0], 1)
    xhat = x_test[xhat_idx]
    
    plt.imshow(xhat.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.savefig(os.path.join('static', 'image.png'))

    # Predict using the model and compare with actual value 
    yhat = model.predict_classes(xhat)
    
    result_str = 'True : ' + str(np.argmax(y_test[xhat_idx[0]])) + ', Predict : ' + str(yhat[0])

    return render_template('index.html', prediction=result_str,
                           img_url=url_for('static', filename='image.png'))

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)
