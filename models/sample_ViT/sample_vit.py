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
        # モデルクラスとパラメータを保存する
        self.model_class = ViT
        self.model_args = dict(
            img_size=img_size,
            patch_size=patch_size,
            n_classes=n_classes,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
        )

        self.optimizer_class = optim.SGD
        self.optimizer_args = dict(lr=0.001, momentum=0.9)
        self.criterion_class = nn.CrossEntropyLoss
        self.criterion_args = dict()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # # モデルの初期化
        # self.net = self.model_class(**self.model_args).to(self.device)
        
        # # 損失関数とオプティマイザの設定
        # self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        # データセット読み込み
        script_dir = os.path.dirname(__file__)
        train_dir = os.path.join(script_dir, "tiny-imagenet-200", "train")
        val_dir = os.path.join(script_dir, "tiny-imagenet-200", "val")

        self.train_dataset = ImageFolder(root=train_dir, transform=self.transform)
        self.val_dataset = ImageFolder(root=val_dir, transform=self.transform)

        # self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.net = None

    def train(self, epochs, mode="single"):
        if mode=="single":
            print("[INFO] Single GPUモードで学習を開始します。")
            self._train_single(epochs)
        elif mode=="distributed":
            world_size = torch.cuda.device_count()
            if world_size <= 1:
                print("[WARNING] 使用可能なGPUが1枚しかありません。Single GPUモードで学習を行います。")
                self._train_single(epochs)
            else:
                print(f"[INFO] Distributedモード ({world_size} GPUs) で学習を開始します。")
                self._train_distributed(epochs, world_size)
                # === 分散学習したモデルを読み込む ===
                self.net = self.model_class(**self.model_args).to(self.device)
                self.net.load_state_dict(torch.load("/tmp/final_model.pth"))
                # self.net.eval()
        else:
            raise ValueError("modeは'single'または'distributed'のいずれかを指定してください。")

    def _train_single(self, epochs):
        print(f"Starting single-device training on {self.device}")

        model = self.model_class(**self.model_args).to(self.device)
        optimizer = self.optimizer_class(model.parameters(), **self.optimizer_args)
        criterion = self.criterion_class(**self.criterion_args)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        model.train()
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0

            # model.train()
            for data in train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()/len(train_loader)
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                epoch_train_acc += acc/len(train_loader)

                del inputs, outputs, loss
            
            print(f'Epoch {epoch+1} : train acc. {epoch_train_acc:.4f}, train loss {epoch_train_loss:.4f}')

        self.net = model

    def _train_distributed(self, epochs, world_size):
        import torch.multiprocessing as mp
        print(f"Starting multi-device training on {world_size} GPUs")
        # mp.spawnを使って各GPUごとにプロセスを立ち上げる
        mp.spawn(self._distributed_worker, args=(epochs, world_size), nprocs=world_size, join=True)

    def _distributed_worker(self, rank, epochs, world_size):
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        # === 1. Initialize the process group ===
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # === 2. set device ===
        device = torch.device(f"cuda:{rank}")

        # === 3. Create model and move it to the appropriate device ===
        model = self.model_class(**self.model_args).to(device)
        model = DDP(model, device_ids=[rank])

        criterion = self.criterion_class(**self.criterion_args)
        optimizer = self.optimizer_class(model.parameters(), **self.optimizer_args)

        # === 4. Create DataLoader with DistributedSampler ===
        script_dir = os.path.dirname(__file__)
        train_dir = os.path.join(script_dir, "tiny-imagenet-200", "train")
        dataset = ImageFolder(root=train_dir, transform=self.transform)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank
        )
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)

        # === 5. Training loop ===
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if rank == 0:
                print(f"[GPU {rank}] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # rank=0だけモデル保存
        if rank == 0:
            torch.save(model.module.state_dict(), "/tmp/final_model.pth")
            
        # === 6. Cleanup ===
        dist.destroy_process_group()

    def infer(self):
        if self.net is None:
            raise RuntimeError("モデルが未学習です。train()を先に呼び出してください。")
        
        self.net.eval()
        criterion = self.criterion_class(**self.criterion_args)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        total_images = len(self.val_dataset)
        print(f"{total_images} 枚の画像を推論するのにかかる時間を計測します。")

        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()/len(val_loader)
                preds = outputs.argmax(dim=1)
                acc = (preds == labels).float().mean()
                total_acc += acc / len(val_loader)

        print(f"推論結果 : acc {total_acc:.4f}, loss {total_loss:.4f}")