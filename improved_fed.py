from typing import List
import torch
import torch.utils.data
import torchvision
from PIL import Image
from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet
from improved_avg import aggregate, dirichlet_loaders

class Configs(BaseConfigs):
    """
    ## Configurations
    """

    device: torch.device = DeviceConfigs()

    # Federated learning setup
    models: List[UNet] = []
    diffusions: List[DenoiseDiffusion] = []
    optimizers: List[torch.optim.Adam] = []
    data_size: List[int] = []
    dataloaders: List[torch.utils.data.DataLoader] = []

    global_model: UNet
    global_diffusion: DenoiseDiffusion

    # Model and training parameters
    image_channels: int = 3
    image_size: int = 32
    n_channels: int = 64
    channel_multipliers: List[int] = [1, 2, 2, 4]
    is_attention: List[bool] = [False, False, False, True]
    n_steps: int = 1_000
    batch_size: int = 64
    n_samples: int = 16
    learning_rate: float = 2e-5
    n_clients: int = 3
    local_iters: int = 1
    epochs: int = 1_000

    dataset: torch.utils.data.Dataset

    def init(self):
        """
        Initialize the configurations.
        """
        self.dataloaders = dirichlet_loaders(self.dataset, self.n_clients, batch_size=self.batch_size)
        self.global_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.global_diffusion = DenoiseDiffusion(
            eps_model=self.global_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        for i in range(self.n_clients):
            # Initialize client models and optimizers
            model = UNet(
                image_channels=self.image_channels,
                n_channels=self.n_channels,
                ch_mults=self.channel_multipliers,
                is_attn=self.is_attention,
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            self.models.append(model)
            self.diffusions.append(
                DenoiseDiffusion(
                    eps_model=model,
                    n_steps=self.n_steps,
                    device=self.device,
                )
            )
            self.optimizers.append(optimizer)
            self.data_size.append(len(self.dataloaders[i]))

        tracker.set_image("sample", True)

    def sample(self):
        """
        Sample images from the global model.
        """
        with torch.no_grad():
            x = torch.randn(
                [self.n_samples, self.image_channels, self.image_size, self.image_size],
                device=self.device,
            )

            for t_ in monit.iterate("Sample", self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.global_diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            tracker.save("sample", x)

    def train(self, idx: int):
        """
        Train the model of a specific client.
        """
        #这里定义了客户端的训练逻辑,次数=客户端的数据量/批次大小
        #其中客户端的数据量是dirichlet_dataloader这个类随机分配的
        #(batch_size,在config中已经写死,为64)
        #因此每个客户端具体进行多少轮训练是不确定的,,,虽然不确定,dirichelet_dataloader()这个类不会让每个客户端分到的数量差的太多
        for data in monit.iterate(f"Train Client {idx + 1}", self.dataloaders[idx]):
            tracker.add_global_step()
            data = data.to(self.device)
            self.optimizers[idx].zero_grad()
            loss = self.diffusions[idx].loss(data)
            loss.backward()
            self.optimizers[idx].step()
            tracker.save("loss", loss)

    # def run(self):
    #     """
    #     Training loop.
    #     """
    #     #当前注释于12-15上传
    #     for epoch in monit.loop(self.epochs): #self.epochs=5表示Server-Clients共进行五次交互   
    #         print(f"Epoch {epoch + 1}/{self.epochs}")
    #         for client_idx in monit.loop(self.n_clients):#self.n_clients=5 表示当前有五个客户端
    #             print(f"  Training Client {client_idx + 1}")
    #             self.models[client_idx].load_state_dict(self.global_model.state_dict())
    #             for _ in range(self.local_iters):#local_iters=1表示当前客户端只进行一次训练,,,注意是一次训练不是一轮训练
    #                 self.train(client_idx)

    #         # Federated averaging  这里是服务器作Fed_avg的逻辑,,主要由fedavg.py中封装的aggregate()类完成
    #         aggregated_dict = aggregate(self.global_model, self.models, self.data_size)
    #         self.global_model.load_state_dict(aggregated_dict)
    #         print("Model aggregation completed")

    #         self.sample()
    #         experiment.save_checkpoint()
    def run(self):
        """
        训练主循环
        """
        for epoch in monit.loop(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            client_losses = []  # 用于存储每个客户端的损失
            for client_idx in monit.loop(self.n_clients):
                print(f"  Training Client {client_idx + 1}")
                self.models[client_idx].load_state_dict(self.global_model.state_dict())
                for _ in range(self.local_iters):
                    self.train(client_idx)

                # 计算客户端的平均损失
                total_loss = 0
                num_batches = 0
                for data in self.dataloaders[client_idx]:
                    data = data.to(self.device)
                    loss = self.diffusions[client_idx].loss(data)  # 使用客户端的 DenoiseDiffusion 计算损失
                    total_loss += loss.item()
                    num_batches += 1
                average_loss = total_loss / num_batches
                client_losses.append(average_loss)  # 将每个客户端的损失存入列表

            # 聚合全局模型，传入损失值
            aggregated_dict = aggregate(self.global_model, self.models, self.data_size, client_losses)
            self.global_model.load_state_dict(aggregated_dict)
            print("  Model aggregation completed")

            # 采样生成图片
            self.sample()
            experiment.save_checkpoint()

    def summary(self):
        """
        Print the summary of the global model.
        """
        print(self.global_model)


class CelebADataset(torch.utils.data.Dataset):
    """
    CelebA HQ dataset
    """

    def __init__(self, image_size: int):
        super().__init__()
        folder = lab.get_data_path() / "celebA"
        self._files = [p for p in folder.glob("**/*.jpg")]
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        img = Image.open(self._files[index])
        return self._transform(img)


@option(Configs.dataset, "CelebA")
def celeb_dataset(c: Configs):
    return CelebADataset(c.image_size)


class MNISTDataset(torchvision.datasets.MNIST):
    """
    MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])
        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


@option(Configs.dataset, "MNIST")
def mnist_dataset(c: Configs):
    return MNISTDataset(c.image_size)


def main():
    experiment.create(name="diffuse", writers={"screen", "labml"})
    configs = Configs()
    experiment.configs(configs, {
        "dataset": "MNIST", #option:MNIST, CelebA
        "image_channels": 1,#1,3
        "epochs": 5,#5,100
        "n_clients": 5,#5,10   上述若干选项第一个是简单版
    })
    configs.init()
    experiment.add_pytorch_models({"eps_model": configs.global_model})

    with experiment.start():
        configs.run()


if __name__ == "__main__":
    main()
