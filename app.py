from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

app = Flask(__name__)

# 모델 로드
model = load_model('your_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # MNIST 데이터셋에서 랜덤한 이미지 선택 
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        rand_index = np.random.randint(0, x_test.shape[0])
        image = x_test[rand_index]
        
        # 이미지 전처리 및 예측 수행 
        image = image.reshape(1, 28, 28)
        prediction = model.predict(image).argmax()

        return render_template('index.html', prediction=prediction,
                               actual=y_test[rand_index],
                               img_data=image.reshape(28, 28))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
