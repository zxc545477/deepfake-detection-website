Deepfake Flask 整合版

資料夾結構：
deepfake_flask_project/
├── app.py
├── model.pth              ← 請把你訓練好的模型放到這裡
├── requirements_flask.txt
├── templates/
│   └── index.html
├── static/
│   └── results/
└── uploads/

使用步驟：
1. 進入資料夾
   cd deepfake_flask_project

2. 安裝套件
   pip install -r requirements_flask.txt

3. 把你訓練好的 model.pth 複製到 deepfake_flask_project 資料夾

4. 執行 Flask
   python app.py

5. 打開瀏覽器
   http://127.0.0.1:5000

注意：
目前 Flask 版支援照片偵測。
影片偵測可作為未來擴充功能。
