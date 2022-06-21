import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from visualize import read_wav_STFT, csv2note


def zxx_cutter(frequency_stamp, time_stamp, Zxx, seg_len, seg_overlap, mode):
    dataloader = []
    start_time = 0
    while start_time + seg_len <= len(time_stamp):
        end_time = min(len(time_stamp), start_time + seg_len)
        if mode == 'data':
            dataloader.append(torch.from_numpy(Zxx[:, start_time:end_time]))
        elif mode == 'label':
            dataloader.append(torch.from_numpy(Zxx[:, start_time + seg_len // 2]))
        start_time += seg_len - seg_overlap
    return dataloader


# https://blog.csdn.net/Eniac0/article/details/117700549
class GoogleNet(nn.Module):
    def __init__(self, num_classes=40, aux_logits=True, init_weight=False):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=(7, 3), stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d((8, 3), stride=(8, 1), ceil_mode=True)  # 当结构为小数时，ceil_mode=True向上取整，=False向下取整
        # nn.LocalResponseNorm （此处省略）
        self.conv2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=(7, 1), padding=1)
        )
        self.maxpool2 = nn.MaxPool2d((6, 3), stride=(4, 1), ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d((6, 3), stride=(4, 1), ceil_mode=True)
        self.conv3 = BasicConv2d(256, 256, kernel_size=(4, 1), stride=(3, 2))
        self.maxpool35 = nn.MaxPool2d((2, 2), ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:  # 使用辅助分类器
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes * 4)

        if init_weight:
            self._initialize_weight()

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        #x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.conv3(x)
        #x = self.inception4a(x)

        x = self.maxpool35(x)

        '''
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        '''

        x = torch.flatten(x, 1)
        x = self.dropout(x)


        x = self.fc(x)
        x = torch.reshape(x, (4, -1))


        #if self.training and self.aux_logits:
            #return x, aux1, aux2
        return x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 创建 Inception 结构函数（模板）
class Inception(nn.Module):
    # 参数为 Inception 结构的那几个卷积核的数量（详细见表）
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 四个并联结构
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 创建辅助分类器结构函数（模板）
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgPool = nn.AvgPool1d(kernel_size=5, stride=3)  # 2d
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14   aux2: N x 528 x 14 x 14（输入）
        x = x.squeeze(3)
        x = self.avgPool(x)
        # aux1: N x 512 x 4 x 4  aux2: N x 528 x 4 x 4（输出） 4 = (14 - 5)/3 + 1
        x = self.conv(x)
        x = torch.flatten(x, 1)     # 展平
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


# 创建卷积层函数（模板）
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


def train(num_epoch, save_path, data_loader, label_loader, device):
    net = GoogleNet(num_classes=40, aux_logits=True, init_weight=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = num_epoch
    best_acc = 0.0
    save_path = save_path
    train_steps = len(data_loader)
    train_loader = data_loader[:int(0.8*train_steps)]
    validate_loader = data_loader[int(0.8*train_steps):]
    train_label = label_loader[:, :int(0.8*train_steps)]
    validate_label = label_loader[:, int(0.8*train_steps):]
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images = data  # labels
            labels = train_label[:, step]
            optimizer.zero_grad()
            logits= net(images.to(device))  # 由于训练的时候会使用辅助分类器，所有相当于有三个返回结果
            print(logits[0].shape)
            print(torch.argmax(labels[0]).shape)
            loss0 = loss_function(logits[0].unsqueeze(0), torch.argmax(labels[0]).unsqueeze(0).to(device).long()) + loss_function(
                logits[1], torch.argmax(labels[1]).to(device).long()) + loss_function(
                logits[2], torch.argmax(labels[2]).to(device).long()) + loss_function(
                logits[3], torch.argmax(labels[3]).to(device).long())

            # loss1 = loss_function(aux_logits1, labels.to(device))
            # loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0  # + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     num_epoch,
                                                                     loss)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for step, val_data in enumerate(val_bar):
                val_images = val_data
                val_labels = validate_label[:, step]
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / len(validate_loader)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    f1, t1, Zxx1 = read_wav_STFT(r'D:\CV\DagstuhlChoirSet_V1.2.3\audio_wav_22050_mono\DCS_LI_FullChoir_Take01_Stereo_STM.wav')
    dataloader1 = zxx_cutter(f1, t1, Zxx1, 5, 2, 'data')
    f2, t2, Zxx2 = read_wav_STFT(
        r'D:\CV\DagstuhlChoirSet_V1.2.3\audio_wav_22050_mono\DCS_LI_FullChoir_Take02_Stereo_STM.wav')
    dataloader2 = zxx_cutter(f2, t2, Zxx2, 5, 2, 'data')
    f3, t3, Zxx3 = read_wav_STFT(
        r'D:\CV\DagstuhlChoirSet_V1.2.3\audio_wav_22050_mono\DCS_LI_FullChoir_Take03_Stereo_STM.wav')
    dataloader3 = zxx_cutter(f3, t3, Zxx3, 5, 2, 'data')
    f = [f1, f2, f3]
    t = [t1, t2, t3]
    dataloader = np.concatenate((dataloader1, dataloader2, dataloader3), axis=0)
    len_data = len(dataloader)
    print(len(dataloader))

    labelloader = []
    for take in range(3):
        num = take + 1
        S_map = csv2note(r'D:\CV\DagstuhlChoirSet_V1.2.3\annotations_csv_scorerepresentation\DCS_LI_FullChoir_Take0{}_Stereo_STM_S.csv'.format(num),
                         'S', f1, t1, 300, -50)
        A_map = csv2note(
            r'D:\CV\DagstuhlChoirSet_V1.2.3\annotations_csv_scorerepresentation\DCS_LI_FullChoir_Take0{}_Stereo_STM_A.csv'.format(num),
            'A', f1, t1, 300, -50)
        T_map = csv2note(
            r'D:\CV\DagstuhlChoirSet_V1.2.3\annotations_csv_scorerepresentation\DCS_LI_FullChoir_Take0{}_Stereo_STM_T.csv'.format(num),
            'T', f1, t1, 300, -50)
        B_map = csv2note(
            r'D:\CV\DagstuhlChoirSet_V1.2.3\annotations_csv_scorerepresentation\DCS_LI_FullChoir_Take0{}_Stereo_STM_B.csv'.format(num),
            'B', f1, t1, 300, -50)
        labelloader1 = [zxx_cutter(f[take], t[take], S_map, 5, 2, 'label'),
                        zxx_cutter(f[take], t[take], A_map, 5, 2, 'label'),
                        zxx_cutter(f[take], t[take], T_map, 5, 2, 'label'),
                        zxx_cutter(f[take], t[take], B_map, 5, 2, 'label')]

        if not len(labelloader):
            labelloader = labelloader1
        else:
            labelloader = np.concatenate((labelloader, labelloader1), axis=1)
        print(np.shape(labelloader))

    train(10, 'model.pth', dataloader, labelloader, 'cuda')