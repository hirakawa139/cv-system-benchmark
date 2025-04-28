import time
from models.sample_ViT.sample_vit import Sample_ViT
from models.sample_ViT.Tiny_ImageNet_downloader import prepare_tiny_imagenet

"""Sample_ViT での学習・推論を呼び出す
"""

class Runner:
    def __init__(self):
        self.model = None
        prepare_tiny_imagenet() # TinyImageNetデータセットがなければダウンロード・解凍
    
    def train(self, epochs=10):
        self.model = Sample_ViT()
        self.model.train(epochs=epochs) # 学習を実行

    def infer(self):
        if self.model is None:
            raise ValueError("モデルが存在しません。学習を行ってから推論を実行してください。")
        
        self.model.infer() # 推論を実行