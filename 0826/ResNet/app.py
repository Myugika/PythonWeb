with open('C:/Workspace/PythonWeb/0826/ResNet/static/ngrok_auth.txt') as nf:
    ngrok_auth = nf.read()


from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='C:/Workspace/PythonWeb/0826/ResNet/static',
                      template_folder='C:/Workspace/PythonWeb/0826/ResNet/templates')
run_with_ngrok(app)

@app.route('/')
def home():
    menu = {'home':1, 'menu':0}
    return render_template('index.html', menu=menu)

@app.route('/menu', methods=['GET','POST'])
def menu():
    menu = {'home':0, 'menu':1}
    if request.method == 'GET':
        languages = [
            {'disp':'영어', 'val':'en'},
            {'disp':'일어', 'val':'jp'},
            {'disp':'중국어', 'val':'cn'},
            {'disp':'프랑스어', 'val':'fr'},
            {'disp':'스페인어', 'val':'es'}
        ]
        return render_template('menu.html', menu=menu,
                                options=languages)   # 서버에서 클라이언트로 정보 전달
    else:
        # 사용자가 입력한 정보를 서버가 읽음
        #index = request.form['index']
        #lang = request.form['lang']
        #lyrics = request.form['lyrics']
        #print(lang, '\n', index, '\n', lyrics, sep='')

        # 사용자가 입력한 파일을 읽어서 upload 디렉토리에 저장
        f_image = request.files['image']
        fname = f_image.filename                # 사용자가 입력한 파일 이름
        filename = os.path.join(app.static_folder, 'upload/') + fname
        f_image.save(filename)

        # 모델 실행
        resnet50 = ResNet50()
        img = Image.open(filename)
        img = img.resize((224,224))
        img_array = np.array(img)
        yhat = resnet50.predict(img_array.reshape(1,224,224,3))
        label = decode_predictions(yhat)
        img_class = label[0][0][1]
        img_prob = label[0][0][2]

        # 모델 실행후 결과를 돌려줌
        result = f'{img_class} ({img_prob*100:.2f}%)'
        mtime = int(os.stat(filename).st_mtime)
        return render_template('menu_res.html', result=result, menu=menu,
                                fname=fname, mtime=mtime)

if __name__ == '__main__':
    app.run()