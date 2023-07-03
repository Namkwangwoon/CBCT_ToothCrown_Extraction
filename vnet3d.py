import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pool, make_kp_layer
from lib.cpool import TopPool, BottomPool, LeftPool, RightPool, FrontPool, BackPool


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        # torch.nn.init.xavier_uniform_(submodule.weight)
        torch.nn.init.normal_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)



class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

class EDUpTransition(nn.Module):
    def __init__(self, inChans, num_classes=4, last_layer=False):
        super(EDUpTransition, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv3d(inChans, inChans//2, kernel_size=3, padding=1)
        self.bn = ContBatchNorm3d(inChans//2)
        self.relu = nn.ReLU()
        self.last_layer = last_layer

        if self.last_layer == True:
            self.last_conv = nn.Conv3d(in_channels=inChans//2, out_channels=num_classes, kernel_size=(1,1,1))

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x):
        # print('decoder input : ', x.shape)
        out = self.upsample(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        if self.last_layer == True:
            out = self.last_conv(out)
            out = F.sigmoid(out)

        # print('decoder out : ', out.shape)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            weight_init_xavier_uniform(m)
    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        return res


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        # self.in_tr = InputTransition(16, elu)
        self.fcn = Conv3DBlock(1, 16)

        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        # self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):
        # out16 = self.in_tr(x)
        # out16 = x
        out16 = self.fcn(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)

        return out

class EDVNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(EDVNet, self).__init__()
        # self.in_tr = InputTransition(16, elu)
        # self.down_tr32 = Conv3DBlock(16, 32)

        # self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = EDUpTransition(256)
        self.up_tr128 = EDUpTransition(128)
        self.up_tr64 = EDUpTransition(64, 16, last_layer=True)
        # self.up_tr32 = EDUpTransition(32, 8, 1, elu)
        # self.out_tr = OutputTransition(32, elu, nll)
        

    def forward(self, x):
        # out32 = self.down_tr32(x)
        x = self.down_tr64(x)
        x = self.down_tr128(x)
        x = self.down_tr256(x)
        out = self.up_tr256(x)
        feat = self.up_tr128(out)
        out = self.up_tr64(feat)
        # out = self.up_tr32(out)

        # out = F.sigmoid(out)

        return feat, out



class InterOutput(nn.Module):
    def __init__(self, inChans, num_classes=4):
        super(InterOutput, self).__init__()

        self.conv = nn.Conv3d(inChans, num_classes, kernel_size=1)

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x):
        out = self.conv(x)
        out = F.sigmoid(out)

        return out



# class EDVNetx3(nn.Module):
#     # the number of convolutions in each layer corresponds
#     # to what is in the actual prototxt, not the intent
#     def __init__(self, elu=True, nll=False):
#         super(EDVNet, self).__init__()
#         # self.in_tr = InputTransition(16, elu)
#         # self.down_tr32 = Conv3DBlock(16, 32)

#         # self.down_tr32 = DownTransition(16, 1, elu)
#         self.down_tr64 = DownTransition(32, 2, elu)
#         self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
#         self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
#         self.up_tr256 = EDUpTransition(256)
#         self.up_tr128 = EDUpTransition(128)
#         self.up_tr64 = EDUpTransition(64, 32, last_layer=True)
#         # self.up_tr32 = EDUpTransition(32, 8, 1, elu)
#         # self.out_tr = OutputTransition(32, elu, nll)
#         self.out1 = InterOutput(128)
#         self.out2 = InterOutput(64)
        

#     def forward(self, x):
#         # out32 = self.down_tr32(x)
#         x = self.down_tr64(x)
#         x = self.down_tr128(x)
#         x = self.down_tr256(x)
#         out1 = self.up_tr256(x)
#         out2 = self.up_tr128(out1)
#         out3 = self.up_tr64(out2)
#         # out = self.up_tr32(out)
#         out1 = self.out1(out1)
#         out2 = self.out2(out2)
#         # out = F.sigmoid(out)

#         return [out1, out2, out3]



class HeatmapVNet(nn.Module):
    def __init__(self):
        super(HeatmapVNet, self).__init__()
        self.backbone = VNet()
        self.encoder_decoder = EDVNet()
        self.box_head = BoxHead()

    def forward(self, x):
        x = self.backbone(x)
        feat, x = self.encoder_decoder(x)
        
        coor1, coor2 = self.box_head(feat)

        return x, coor1, coor2
    
    
# class HeatmapVNetx3(nn.Module):
#     def __init__(self):
#         super(HeatmapVNet, self).__init__()
#         self.backbone = VNet()
#         self.encoder_decoder = EDVNetx3()

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.encoder_decoder(x)

#         return x
    
    

class BoxHead(nn.Module):
    def __init__(self):
        super(BoxHead, self).__init__()
        
        # self.pooling_module_1 = pool(64, TopPool, LeftPool, BackPool)
        # self.pooling_module_2 = pool(64, BottomPool, RightPool, FrontPool)
        
        self.regression_module_1 = make_kp_layer(64, 64, 3)
        self.regression_module_2 = make_kp_layer(64, 64, 3)
        
    def forward(self, feat):
        '''
        feat : [b, c, d, h, w]
        ind : [b, num_objects]
        '''
        
        # # x1 = self.pooling_module_1(feat)
        # # x2 = self.pooling_module_2(feat)
        
        x1 = self.regression_module_1(feat)
        x2 = self.regression_module_2(feat)
        
        return x1, x2