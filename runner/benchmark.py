import os
import json
from datetime import datetime
import argparse
import torch
from runner.timer import Timer
from models.model_runner import load_model_runner

RUNNER_DIR = os.path.abspath(os.path.dirname(__file__)) 
LOG_DIR = os.path.join(RUNNER_DIR, "..", "logs")  

def benchmark_model(model_name, epochs):
    model_runner = load_model_runner(model_name)
    if model_runner is None:
        return
    
    print(f"==== {model_name} の学習を開始 ====")
    train_timer = Timer()
    train_timer.start()
    model_runner.train(epochs)
    train_time = train_timer.stop()
    print(f"学習時間: {train_time:.2f} 秒")

    print(f"==== {model_name} の推論を開始 ====")
    infer_timer = Timer()
    infer_timer.start()
    model_runner.infer()
    infer_time = infer_timer.stop()
    print(f"推論時間: {infer_time:.2f} 秒")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "epochs": epochs,
        "train_time": f"{train_time:.2f} sec",
        "infer_time": f"{infer_time:.2f} sec",
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{model_name}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"==== ベンチマーク完了: {log_path} に結果を保存 ====")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="深層学習モデルのベンチマークツール")
    parser.add_argument("model", type=str, help="使用するモデルの名前(models/ のサブディレクトリ名)")
    parser.add_argument("--epochs", type=int, default=10, help="学習を行う epoch 数")
    args = parser.parse_args()

    benchmark_model(args.model, args.epochs)
