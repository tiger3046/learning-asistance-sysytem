import keras
import sys, os
import numpy as np
from keras.models import load_model
from PIL import Image
import requests
import glob
import re
import shutil
import oo
import time


#クラスstudyingは機械学習済みファイルを用いた画像判定を行う
class studying:



    def __init__(self,testpic,keras_param):
        self.testpic = testpic
        self.keras_param = keras_param
        self.imsize = (64, 64)


    def load_image(self,path):
        img = Image.open(path)
        img = img.convert('RGB')
        # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
        img = img.resize(self.imsize)
        # 画像データをnumpy配列の形式に変更
        img = np.asarray(img)
        img = img / 255.0
        return img

    def ai_judge(self):
        model = load_model(self.keras_param)
        img = self.load_image(self.testpic)
        prd = model.predict(np.array([img]))
        print(prd) # 精度の表示
        prelabel = np.argmax(prd, axis=1)


        if prelabel == 0:
            return 0
        elif prelabel == 1:
            return 1

def main():
    send_line_notify(msg)
    send_line_notify(msg1)


def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'LINEnotifyのトークンをいれてください'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)


m = 0 #15回判定のためのカウンタ変数
file = "./cnn.h5" #機械学習済みファイルを指定してください
src_file = glob.glob('./photofile/*.jpg')#写真ファイルを指定してください
datacount = len(src_file)
count = 0
un_count = 0
s = 2

while s > 0:
    if len(src_file) == 15:
        for i in src_file:
            a = studying(i,file)
            if a.ai_judge() == 0:
                count += 1
                m += 1
            elif a.ai_judge() == 1:
                un_count += 1
                m += 1
        if m == 15:
            oo.allremove()
            break
    else:
        src_file = glob.glob('./photofile/*.jpg')
        continue




#判定結果に合わせて内容をかえる

if count > un_count:
    msg = 'おつかれ'
else:
    msg = 'みんな勉強してるよ'
    msg1 = '君だけやってないの大丈夫？'



#lineの出力
if __name__ == "__main__":
    main()
