from archs.resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    def __init__(self, n_segment=3, n_div=8, inplace=False, second_segments=2):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.second_segments = second_segments
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x, unlabeled):
        if unlabeled:
            #print("using unlabeled shift")
            x = self.shift(x,self.second_segments,
                           fold_div=self.fold_div, inplace=self.inplace)
        else:
            #print(x.size())
            x = self.shift(x, self.n_segment,
                           fold_div=self.fold_div, inplace=self.inplace)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        #print("segment_size is {}".format(n_segment))
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            #raise NotImplementedError
            out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-
                                           1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None



class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        raise NotImplementedError
        # nt, c, h, w = x.size()
        # n_batch = nt // n_segment
        # x = x.view(n_batch, n_segment, c, h, w).transpose(
        #     1, 2)  # n, c, t, h, w
        # x = F.max_pool3d(x, kernel_size=(3, 1, 1),
        #                  stride=(2, 1, 1), padding=(1, 0, 0))
        # x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        # return x


class make_block_temporal(nn.Module):
    def __init__(self, stage, this_segment=3,n_div=8, second_segments=2):
        super(make_block_temporal, self).__init__()
        self.blocks = nn.ModuleList(list(stage.children()))
        self.second_segments = second_segments
        print('=> Processing stage with {} blocks'.format(len(self.blocks)))
        self.temporal_shift = TemporalShift(n_segment=this_segment, n_div=n_div, second_segments= self.second_segments)
#        for i, b in enumerate(self.blocks):
 #           self.blocks[i]= nn.Sequential(b)
    def forward(self,x,unlabeled=False):
        for i, b in enumerate(self.blocks):
            x= self.temporal_shift(x,unlabeled)
            x = self.blocks[i](x)
        return x

class make_blockres_temporal(nn.Module):
    def __init__(self, stage, this_segment=3,n_div=8, n_round=1, second_segments=2):
        super(make_blockres_temporal, self).__init__()
        self.blocks = nn.ModuleList(list(stage.children()))
        self.second_segments = second_segments
        self.n_round = n_round
        print('=> Processing stage with {} blocks'.format(len(self.blocks))) 
        self.temporal_shift = TemporalShift(n_segment=this_segment, n_div=n_div, second_segments=self.second_segments)
   #     for i, b in enumerate(self.blocks):
  #          self.blocks[i]= nn.Sequential(b)
 #           print(self.blocks[i])

    def forward(self,x,unlabeled=False):
        #print("make_block_res_temporal_called")
        for i, b in enumerate(self.blocks):
            #print(x.size())
            if i% self.n_round == 0:
                #print("size of x is {}".format(x.size()))
                x= self.temporal_shift(x,unlabeled)
            x = self.blocks[i](x)
        return x


def make_temporal_shift(net, n_segment,second_segments=2, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment //
                          2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, ResNet):
        if place == 'block':
            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], n_div, second_segments)
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], n_div, second_segments)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], n_div, second_segments)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], n_div, second_segments)

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            net.layer1 = make_blockres_temporal(net.layer1, n_segment_list[0], n_div, n_round, second_segments)
            net.layer2 = make_blockres_temporal(net.layer2, n_segment_list[1], n_div, n_round, second_segments)
            net.layer3 = make_blockres_temporal(net.layer3, n_segment_list[2], n_div, n_round, second_segments)
            net.layer4 = make_blockres_temporal(net.layer4, n_segment_list[3], n_div, n_round, second_segments)
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')
