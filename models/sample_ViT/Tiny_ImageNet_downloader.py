import os 
import zipfile
import urllib.request
import shutil

"""Tiny ImageNetのダウンロードと前処理を行うスクリプト
"""

def download_tiny_imagenet(zip_path="tiny-imagenet-200.zip", data_dir="tiny-imagenet-200"):
    """
    Download the Tiny ImageNet dataset if it is not already present.
    
    Parameters:
    zip_path (str): Path to save the downloaded zip file.
    url (str): URL to download the dataset from.
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Tiny ImageNet dataset already downloaded.")

    if not os.path.exists(data_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(os.path.abspath(__file__)))
        print("Extraction complete.")
    else:
        print("Tiny ImageNet dataset already extracted.")

def convert_val_to_imagefolder_format(data_dir="tiny-imagenet-200"):
    """
    Convert the validation data to ImageFolder format.
    
    Parameters:
    data_dir (str): Path to the Tiny ImageNet dataset directory.
    """
    val_dir = os.path.join(data_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")

    print(f"Converting validation data in {val_dir} to ImageFolder format...")

    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name = parts[0]
            label = parts[1]

            label_dir = os.path.join(val_dir, "val", label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            src_path = os.path.join(images_dir, img_name)
            dst_path = os.path.join(label_dir, img_name)
            shutil.move(src_path, dst_path)

    # 不要になったフォルダ・ファイルを削除
    shutil.rmtree(images_dir)
    os.remove(annotations_file)

    print("Validation data conversion complete.")

def prepare_tiny_imagenet():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(BASE_DIR, "tiny-imagenet-200.zip")
    data_dir = os.path.join(BASE_DIR, "tiny-imagenet-200")

    download_tiny_imagenet(zip_path, data_dir)
    convert_val_to_imagefolder_format(data_dir)

if __name__ == "__main__":
    prepare_tiny_imagenet()