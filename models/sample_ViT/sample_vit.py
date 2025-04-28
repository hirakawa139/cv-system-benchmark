import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.sample_ViT.vit_model import ViT
import os

class Sample_ViT:
    def __init__(self, img_size=64, patch_size=8, n_classes=200, dim=512,
        depth=6, n_heads=8, mlp_dim=512, batch_size=100):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        # モデルの初期化
        self.net = ViT(
            img_size=img_size,
            patch_size=patch_size,
            n_classes=n_classes,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
        ).to(self.device)

        # 損失関数とオプティマイザの設定
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        # データセットの準備
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        script_dir = os.path.dirname(__file__)
        train_dir = os.path.join(script_dir, "tiny-imagenet-200", "train")
        val_dir = os.path.join(script_dir, "tiny-imagenet-200", "val")

        self.train_dataset = ImageFolder(root=train_dir, transform=transform)
        self.val_dataset = ImageFolder(root=val_dir, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def train(self, epochs):
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            epoch_test_loss = 0.0
            epoch_test_acc = 0.0

            self.net.train()
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()/len(self.train_loader)
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                epoch_train_acc += acc/len(self.train_loader)

                del inputs
                del outputs
                del loss

            self.net.eval()
            with torch.no_grad():
                for data in self.val_loader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    epoch_test_loss += loss.item()/len(self.val_loader)
                    test_acc = (outputs.argmax(dim=1) == labels).float().mean()
                    epoch_test_acc += test_acc/len(self.val_loader)

            print(f'Epoch {epoch+1} : train acc. {epoch_train_acc:.4f}, train loss {epoch_train_loss:.4f}')
            print(f'Epoch {epoch+1} : test acc. {epoch_test_acc:.4f}, test loss {epoch_test_loss:.4f}')

    def infer(self):
        self.net.eval()
        total_images = len(self.val_loader.dataset)
        print(f"{total_images} 枚の画像を推論するのにかかる時間を計測します。")
        
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)