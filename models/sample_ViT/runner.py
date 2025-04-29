import time
from models.sample_ViT.sample_vit import Sample_ViT
from models.sample_ViT.Tiny_ImageNet_downloader import prepare_tiny_imagenet

"""Sample_ViT での学習・推論を呼び出す
"""

class Runner:
    def __init__(self):
        prepare_tiny_imagenet() # TinyImageNetデータセットがなければダウンロード・解凍
        self.model = Sample_ViT()
    
    def train(self, epochs=10, mode="single"):
        self.model.train(epochs=epochs, mode=mode) # 学習を実行
        self.model_is_trained = True

    def infer(self):
        self.model.infer() # 推論を実行