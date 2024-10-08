import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.nn import AdaptiveAvgPool3d
from torch.nn import Linear
from channel_exchange.models.modules import BatchNorm3dParallel, ModuleParallel
from channel_exchange.models.modules import Exchange

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def generate_model(model_depth, in_channels, num_cls_classes, pretrain_path=None):


    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet10(
            # sample_input_W=input_W,
            # sample_input_H=input_H,
            # sample_input_D=input_D,
            num_cls_classes=num_cls_classes,
            shortcut_type="B",
            in_channels=in_channels,
            enc_features=512).cuda()
    elif model_depth == 18:
        model = resnet18(
            # sample_input_W=input_W,
            # sample_input_H=input_H,
            # sample_input_D=input_D,
            num_cls_classes=num_cls_classes,
            shortcut_type="A",
            in_channels=in_channels,
            enc_features=512)
    elif model_depth == 34:
        model = resnet34(
            # sample_input_W=input_W,
            # sample_input_H=input_H,
            # sample_input_D=input_D,
            num_cls_classes=num_cls_classes,
            shortcut_type="A",
            in_channels=in_channels,
            enc_features=512)
    elif model_depth == 50:
        model = resnet50(
            # sample_input_W=input_W,
            # sample_input_H=input_H,
            # sample_input_D=input_D,
            num_cls_classes=num_cls_classes,
            shortcut_type="B",
            in_channels=in_channels,
            enc_features=2048)
    elif model_depth == 101:
        model = resnet101(
            # sample_input_W=input_W,
            # sample_input_H=input_H,
            # sample_input_D=input_D,
            num_cls_classes=num_cls_classes,
            shortcut_type="B",
            in_channels=in_channels,
            enc_features=2048)
    elif model_depth == 152:
        model = resnet152(
            # sample_input_W=input_W,
            # sample_input_H=input_H,
            # sample_input_D=input_D,
            num_cls_classes=num_cls_classes,
            shortcut_type="B",
            in_channels=in_channels,
            enc_features=2048)
    elif model_depth == 200:
        model = resnet200(
            # sample_input_W=input_W,
            # sample_input_H=input_H,
            # sample_input_D=input_D,
            num_cls_classes=num_cls_classes,
            shortcut_type="B",
            in_channels=in_channels,
            enc_features=2048)

    net_dict = model.state_dict()

    # load pretrain
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, weights_only=True)

        conv1_weights = pretrain["state_dict"]["module.conv1.weight"].repeat(1,in_channels,1,1,1)
        pretrain["state_dict"]["module.conv1.weight"] = conv1_weights

        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        # new_parameters = []
        # for pname, p in model.named_parameters():
        #     for layer_name in new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break
        #
        # new_parameters_id = list(map(id, new_parameters))
        # base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        # parameters = {'base_parameters': base_parameters,
        #               'new_parameters': new_parameters}

        # return model, parameters
        return model

    return model #, model.parameters()


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return ModuleParallel(nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False))


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data.cuda(), zero_pads.cuda()], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = BatchNorm3dParallel(planes, 2)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = BatchNorm3dParallel(planes, 2)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = ModuleParallel(nn.Conv3d(inplanes, planes, kernel_size=1, bias=False))
        self.bn1 = BatchNorm3dParallel(planes, 2)
        self.conv2 = ModuleParallel(nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False))
        self.bn2 = BatchNorm3dParallel(planes, 2)
        self.conv3 = ModuleParallel(nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False))
        self.bn3 = BatchNorm3dParallel(planes * 4, 2)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        self.exchange = Exchange()
        self.bn_threshold = 2e-2
        # assert self.bn_threshold != None

        self.num_parallel = 2
        # assert self.num_parallel != None

        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm3d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 # sample_input_D,
                 # sample_input_H,
                 # sample_input_W,
                 num_cls_classes=2,
                 shortcut_type='B',
                 in_channels=1,
                 enc_features=512,
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = ModuleParallel(nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False))

        self.bn1 = BatchNorm3dParallel(64, 2)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.avgpool = ModuleParallel(AdaptiveAvgPool3d(output_size=(1, 1, 1)))
        self.fc = ModuleParallel(Linear(in_features=enc_features, out_features=num_cls_classes, bias=True))
        

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    ModuleParallel(nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False)), BatchNorm3dParallel(planes * block.expansion, 2))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # x = x.view(x.size(0), -1)    
        x = [tensor.view(tensor.size(0), -1) for tensor in x] 

        x = self.fc(x)

        ens = 0
        alpha_soft = F.softmax(torch.tensor([1.0, 1.0]), dim=0)
        for l in range(2):
            ens += alpha_soft[l] * x[l].detach()        
        
        x.append(ens)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model