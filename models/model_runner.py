import importlib

def load_model_runner(model_name):
    """指定されたモデルの学習・推論用モジュールをロード

    Args:
        model_name (string): ロードしたいモデルのディレクトリ名

    Returns:
        _type_: ロードされたモジュール
    """    
    try:
        model_module = importlib.import_module(f"models.{model_name}.runner")
        return model_module
    except ModuleNotFoundError:
        print(f"モデル {model_name} の runner が見つかりません")
        return None