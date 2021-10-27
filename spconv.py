import torch
import torch.nn.functional as F
import torch.nn as nn
import torch


def generate_indices(value, mask):
    A = torch.arange(3).view(-1, 1, 1)
    # self.spa_mask: torch.Size([1, 1, 256, 256])
    mask_indices = torch.nonzero(mask.squeeze())  # [num, 2]

    # indices: dense to sparse (1x1)
    h_idx_1x1 = mask_indices[:, 0]
    w_idx_1x1 = mask_indices[:, 1]

    # indices: dense to sparse (3x3)
    mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A  # [num, 2]-->[3, num, 2]

    h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)
    w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)

    # indices: sparse to sparse (3x3)
    #indices = torch.arange(float(mask_indices.size(0))).view(1, -1) + 1

    #mask[0, 0, h_idx_1x1, w_idx_1x1] = indices.long()
    #idx_s2s = F.pad(mask, [1, 1, 1, 1])[0, :, h_idx_3x3, w_idx_3x3].view(9, -1).long()

    return h_idx_3x3, w_idx_3x3

def mask_select(x, k, h_idx_3x3, w_idx_3x3):
    #h_idx_3x3:torch.Size([294480])
    #w_idx_3x3:torch.Size([294480])
    #F.pad(x, [1, 1, 1, 1]):torch.Size([1, 64, 258, 258])
    #F.pad(x, [1, 1, 1, 1])[0, :, h_idx_3x3, w_idx_3x3]: torch.Size([64, 294723])
    #result [576, 32660])
    result = F.pad(x, [1, 1, 1, 1])[0, :, h_idx_3x3, w_idx_3x3].view(9 * x.size(1), -1)
    return result

def spar2den(fea, mask, input):
    result = torch.zeros_like(input)
    mask_indices = torch.nonzero(mask.squeeze())  # [num, 2]

    h_idx_1x1 = mask_indices[:, 0]
    w_idx_1x1 = mask_indices[:, 1]
    input[0, :, h_idx_1x1, w_idx_1x1] = fea
    return input

if __name__ == '__main__':
    input = torch.randn((1, 64, 256, 256))
    mask = torch.randint(2, (1, 1, 256, 256))

    weight = torch.randn((3, 3, 64, 64))

    h_idx_3x3, w_idx_3x3 = generate_indices(input, mask)

    ############## normal convolution#############
    fea_col = F.unfold(input, 3, stride=1, padding=1).squeeze(0)
    fea_d2d = torch.mm(weight.view(64, -1), fea_col)
    fea_d2d = fea_d2d.view(1, 64, 256, 256)


    ##############  sparse convolution##############
    fea_d2s = torch.mm(weight.view(64, -1), mask_select(input, 3, h_idx_3x3, w_idx_3x3))  # [64, 32680]
    fea_d2s_reshape = spar2den(fea_d2s, mask, input)


    # ############### sparse to dense & sparse###############
    # fea_s2ds = torch.mm(weight.view(64, -1), F.pad(fea_d2s, [1, 0, 0, 0])[:, idx_s2s].view(64 * 3 * 3, -1))



#############check############
# if __name__ == '__main__':
#     input = torch.arange(12).reshape(1,1,3,4).float()
#     print(input)
#
#     mask = torch.Tensor([[[[0, 1, 0, 1],
#                             [1, 0, 1, 0],
#                             [0, 1, 0, 1]]]])
#
#     weight = torch.ones((3,3,1,1))
#
#     h_idx_3x3, w_idx_3x3, idx_s2s = generate_indices(input, mask)
#     ############## normal convolution#############
#     fea_col = F.unfold(input, 3, stride=1, padding=1).squeeze(0)
#     fea_d2d = torch.mm(weight.view(1, -1), fea_col)
#     fea_d2d = fea_d2d.view(1, 1, 3, 4)
#     print(fea_d2d)
#
#     ##############  sparse convolution##############
#     fea_d2s = torch.mm(weight.view(1, -1), mask_select(input, 3, h_idx_3x3, w_idx_3x3))  # [64, 32680]
#     fea_d2s_reshape = spar2den(fea_d2s, mask, input)
#     print(fea_d2s_reshape)
'''
input tensor
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.]]]])
          
normal convolution result (assuming conv weights are all one)
tensor([[[[10., 18., 24., 18.],
          [27., 45., 54., 39.],
          [26., 42., 48., 34.]]]])
          
sparse conv result(mask)
tensor([[[[ 0., 18.,  2., 18.],
          [27.,  5., 54.,  7.],
          [ 8., 42., 10., 34.]]]]
'''