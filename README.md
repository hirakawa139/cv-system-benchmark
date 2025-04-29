# cv-system-benchmark

## 実行環境
python: 3.10.14

## 使用方法
1. ソースコードをクローン
```
git clone https://github.com/hirakawa139/cv-system-benchmark.git
```

2. 必要なライブラリをインストールする
```
pip install -r requirements.txt
```

3. 以下のようにしてsampleViTでのベンチマークを実行
```
python runner/benchmark.py  sampleViT --epochs=10 --mode=single
```
※mode：single or distributed を指定
