import time
import torch

class Timer:
    """時間計測を一元管理するモジュール
    """    
    def __init__(self, use_gpu=True):
        """GPUを使うか(あるいは使えるか)に合わせて初期化

        Args:
            use_gpu (bool, optional): GPUを使用するかどうか. Defaults to True.
        """        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None
            self.end_time = None
    
    def start(self):
        """タイマー開始
        """        
        if self.use_gpu:
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_time = time.time()

    def stop(self):
        """タイマーストップ

        Returns:
            _type_: 計測結果(時間:秒)
        """        
        if self.use_gpu:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000 # 秒単位に直している
        else:
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
        return elapsed_time