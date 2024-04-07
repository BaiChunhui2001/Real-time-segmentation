import torch
import torch.nn as nn
import torch.nn.functional as F


# def avg_reduce_channel(x):
#     # Reduce channel by avg
#     # Return cat([avg_ch_0, avg_ch_1, ...])
#     if not isinstance(x, (list, tuple)):
#         return torch.mean(x, dim=1, keepdim=True)
#     elif len(x) == 1:
#         return torch.mean(x[0], dim=1, keepdim=True)
#     else:
#         res = []
#         for xi in x:
#             res.append(torch.mean(xi, dim=1, keepdim=True))
#         return torch.cat(res, dim=1)

# def avg_reduce_hw(x):
#     # Reduce hw by avg
#     # Return cat([avg_pool_0, avg_pool_1, ...])
#     if not isinstance(x, (list, tuple)):
#         return F.adaptive_avg_pool2d(x, 1)
#     elif len(x) == 1:
#         return F.adaptive_avg_pool2d(x[0], 1)
#     else:
#         res = []
#         for xi in x:
#             res.append(F.adaptive_avg_pool2d(xi, 1))
#         return torch.cat(res, dim=1)

def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    # print("mean_value: ", mean_value)
    # print("max_value: ", max_value)

    if use_concat:
        res = torch.cat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res

def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.cat(res, dim=1)

def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when dim=[2, 3], the paddle.max api has bug for training.
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = F.adaptive_max_pool2d(x, 1)

    if use_concat:
        res = torch.cat([avg_pool, max_pool], dim=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return torch.cat(res, dim=1)


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class ConvBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out

class ConvBNAct(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, act_type="leakyrelu"):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if act_type == "leakyrelu":
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_out = ConvBNReLU(
            in_ch, out_ch, kernel=3)
        
    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        out = self.fuse(x, y)
        return out


class UAFM_ChAtten(UAFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch)

        self.conv_xy_atten = nn.Sequential(
            ConvBNAct(
                2 * in_ch,
                in_ch // 4,
                kernel=1,
                act_type="leakyrelu"),
            ConvBN(in_ch // 4, in_ch // 2, kernel=1))



    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_hw([x, y], self.training)
        atten = torch.sigmoid(self.conv_xy_atten(atten))

        out = torch.cat([x * atten ,y * (1 - atten)],dim=1)
        out = self.conv_out(out) + torch.cat([x ,y],dim=1)
        return out
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params




class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel=3),
            ConvBN(
                2, 1, kernel=3))
        
        self.init_weight()

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        # print(atten.shape)
        atten = torch.sigmoid(self.conv_xy_atten(atten))

        out =torch.cat([x * atten ,y * (1 - atten)],dim=1)
        out = self.conv_out(out)
        return out
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
    
if __name__ == "__main__":
    net = UAFM_ChAtten(256,256)
    x = torch.randn(16, 128, 128, 128)
    y = torch.randn(16, 128, 128, 128)
    out = net(x,y)
    print(out.size())
   
    # net.get_params()


