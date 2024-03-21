import torch

from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv, SPPF, CSPSPPF_ori, CSPSPPF_group, C3, get_norm, get_act
from SimAM import C2f_SimAM, C3_SimAM

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# c_in = c_out = 1024
#
# # Convolutional layer
# conv1 = MixConv2d(c_in, c_out, (1, 3, 5, 7), 1)
# conv2 = MixConv2d(c_in, c_out, (3, 5, 7, 9), 1)
# conv3 = Conv(c_in, c_out, 3, 1)
# results1 = profile(input=torch.randn(1, c_in, 80, 80), ops=[conv1, conv2, conv3], n=10)
#
# # Spatial pyramid pooling module
# s1 = SPPF(c_in, c_out, k=3)
# s2 = CSPSPPF_group(c_in, c_out, k=5)
# s3 = CSPSPPF_ori(c_in, c_out, k=7)
# results2 = profile(input=torch.randn(1, c_in, 80, 80), ops=[s1, s2, s3], n=10)
#
# # C2f_SimAM vs C3_SimAM
# sim1 = C2f_SimAM(c_in, c_out, n=1, shortcut=True, g=1, e=0.5)   # v8
# sim2 = C3_SimAM(c_in, c_out, n=1, shortcut=True, g=1, e=0.5)    # v5
# sim3 = C3(c_in, c_out)
# result3 = profile(input=torch.randn(1, c_in, 80, 80), ops=[sim1, sim2, sim3], n=10)

input = torch.randn(4, 1024, 80, 80)
norm = get_norm('bn_2d')
print(norm)
