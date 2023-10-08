from flask import Flask, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import random

app = Flask(__name__)

# 모델 불러오기
model = load_model('model.h5')

@app.route('/predict', methods=['GET'])
def predict():
    # MNIST 데이터셋 로드하기 (학습 데이터는 필요 없으므로 버립니다)
    (_, _), (x_test, y_test) = mnist.load_data()

    # 테스트 데이터셋에서 임의로 하나 선택하기
    idx = random.randint(0, len(x_test) - 1)
    image = x_test[idx]
    
    # 이미지 전처리하기 (모델 입력에 맞게 차원 추가 및 정규화)
    input_image = image.reshape(-1, 28, 28) / 255.

    # 모델로 예측하기 
    yhat = model.predict(input_image)

     # 가장 확률이 높은 클래스 선택하기 
    predicted_class = np.argmax(yhat[0])

    return jsonify({'prediction': int(predicted_class), 'truth': int(y_test[idx])})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
