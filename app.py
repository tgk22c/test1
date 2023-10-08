from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

app = Flask(__name__)

# Load pre-trained model when the app starts.
model = load_model('model.h5')

# Load the MNIST dataset.
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Select a random test image and its label.
        i = np.random.randint(0, len(x_test))
        image, true_label = x_test[i], y_test[i]

        # Predict the digit in the image.
        pred_label = np.argmax(model.predict(image.reshape(1, -1)), axis=-1)

        return render_template('index.html', pred_label=pred_label[0], true_label=true_label)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
