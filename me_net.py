import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
try:
    from module import *
except:
    from .module import *


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        # bottleneck_output = conv(relu(norm(concated_features)))
        bottleneck_output = conv(concated_features)
        return bottleneck_output

    return bn_function


# class _DenseLayer(nn.Module):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
#         super(_DenseLayer, self).__init__()
#         self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
#         self.add_module('relu1', nn.ReLU(inplace=True)),
#         self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
#                         kernel_size=1, stride=1, bias=False)),
#         self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
#         self.add_module('relu2', nn.ReLU(inplace=True)),
#         self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
#                         kernel_size=3, stride=1, padding=1, bias=False)),
#         self.drop_rate = drop_rate
#         self.efficient = efficient
#
#     def forward(self, *prev_features):
#         bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
#         if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
#             bottleneck_output = cp.checkpoint(bn_function, *prev_features)
#         else:
#             bottleneck_output = bn_function(*prev_features)
#         new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         return new_features

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
        #                 kernel_size=1, stride=1, bias=False)),
        # self.add_module('norm2', nn.BatchNorm2d(growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),

        # self.add_module('Pinwheel_shapedConv1', Pinwheel_shapedConv(c1=num_input_features, c2=bn_size * growth_rate, k=3, s=1)),
        # self.add_module('Pinwheel_shapedConv2', Pinwheel_shapedConv(c1=bn_size * growth_rate, c2=growth_rate, k=3, s=1)),

        self.add_module('Conv2d', nn.Conv2d(num_input_features, growth_rate,kernel_size=1)),
        self.add_module('WTConv2d_Res', WTConv2d_Res(growth_rate, growth_rate)),

        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.Conv2d)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.WTConv2d_Res(bottleneck_output)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
        self.add_module('meanpool', nn.AvgPool2d(kernel_size=3, stride=2,padding=1))

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x1 = self.maxpool(x)
        x2 = self.meanpool(x)

        x = x1 + x2
        return x

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class OursNet4(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,img_size=224,
                 num_classes=10, small_inputs=False, efficient=False):

        super(OursNet4, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.dense_blocks.append(block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.transitions.append(trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.norm_final = nn.BatchNorm2d(num_features)

        # self.rga = RGA_Module(in_channel=num_features,
        #                       in_spatial=self.calculate_downsampled_size(img_size, len(block_config) + 1) ** 2)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # 中间层损失
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_init_features + block_config[0] * growth_rate, num_classes)
        )

        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int((num_init_features + block_config[0] * growth_rate) * compression) + block_config[1] * growth_rate, num_classes)
        )

        self.classifier3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int((int((num_init_features + block_config[0] * growth_rate)* compression) + block_config[1] * growth_rate)* compression) + block_config[2] * growth_rate, num_classes)
        )

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                # pass
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

        self.id = nn.Identity()

    def forward(self, x):
        x = self.features(x)

        outputs = []
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            # outputs.append(x)
            if i==0:
                outputs.append(self.classifier1(x))
            elif i==1:
                outputs.append(self.classifier2(x))
            elif i==2:
                outputs.append(self.classifier3(x))

            if i < len(self.transitions):
                x = self.transitions[i](x)

        out = self.norm_final(x)
        out = F.relu(out, inplace=True)

        # out = self.rga(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.id(out)
        out = self.classifier(out)
        return out, outputs
        # return out

    # initial_size = 224
    # num_downsamples = 5
    # result_size = calculate_downsampled_size(initial_size, num_downsamples)
    def calculate_downsampled_size(self, initial_size, num_downsamples):
        size = initial_size // 2
        for _ in range(1, num_downsamples):
            size //= 2
        return size


if __name__ == '__main__':
    model = OursNet4(growth_rate=32, block_config=(2, 4, 6, 8), num_init_features=64)
    print(model)

