## This code is about the sparse convolution operation based on pytorch instead of CUDA


## Usage
1.You need to input a tensor like 

# torch.randn((1, 64, 256, 256))(only support batchsize 1)

2.input a mask like: 

# mask = torch.randint(2, (1, 1, 256, 256))

3.a convolution weight like:

# weight = torch.randn((3, 3, 64, 64))


## Verify
'''
input tensor


tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.]]]])

mask tensor


tensor([[[[0, 1, 0, 1],
          [1, 0, 1, 0],
          [0, 1, 0, 1]]]])
          
normal convolution result (assuming conv weights are all one)


tensor([[[[10., 18., 24., 18.],
          [27., 45., 54., 39.],
          [26., 42., 48., 34.]]]])
          
sparse conv result(mask)


tensor([[[[ 0., 18.,  2., 18.],
          [27.,  5., 54.,  7.],
          [ 8., 42., 10., 34.]]]]
'''



