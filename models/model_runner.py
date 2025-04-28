import importlib
import os
import sys

def load_model_runner(model_name):
    """指定されたモデルの学習・推論用モジュールをロード

    Args:
        model_name (string): ロードしたいモデルのディレクトリ名

    Returns:
        _type_: ロードされたモジュール
    """    
    try:
        # プロジェクトのルートディレクトリを sys.path に追加
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        # デバッグ用の出力
        print(f"sys.path: {sys.path}")
        print(f"Attempting to import: models.{model_name}.runner")
        model_module = importlib.import_module(f"models.{model_name}.runner")
        return model_module
    except ModuleNotFoundError:
        print(f"モデル {model_name} の runner が見つかりません")
        return None