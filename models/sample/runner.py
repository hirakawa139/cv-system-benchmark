import time
from models.sample.train_vit import train_vit

"""時間計測のみをテストするサンプルコード
"""

def train(epochs=10):
    train_vit(epochs=epochs)

def infer():
    time.sleep(5)