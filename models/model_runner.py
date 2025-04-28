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
        # print(f"sys.path: {sys.path}")
        # print(f"Attempting to import: models.{model_name}.runner")
        model_module = importlib.import_module(f"models.{model_name}.runner")

        # Runnerクラスが存在するか確認
        if not hasattr(model_module, "Runner"):
            print(f"モデル {model_name} の runner に Runner クラスが定義されていません。")
            return None
        
        return model_module
    
    except ModuleNotFoundError:
        print(f"モデル {model_name} の runner が見つかりません")
        return None
    except Exception as e:
        print(f"モデル {model_name} のロード中にエラーが発生しました: {e}")
        return None