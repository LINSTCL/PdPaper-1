import numpy as np
import torch
import paddle

pData = np.load('tresnet_m.pdparams', allow_pickle=True)
tData = torch.load('tresnet_m.pth')
print(tData['model']["head.fc.weight"])
# del pData['StructuredToParameterName@@']

# paddle_dict = {}
# # for temp in pData.keys():
# #     paddle_dict[temp] = tData['model'][temp.replace()]

# for temp in zip(pData.keys(), tData['model'].keys()):
#     if temp[0] == "head.fc.weight":
#         print(tData['model'][temp[1]].permute((1,0)).shape)
#         print(tData['model'][temp[1]])
#         paddle_dict[temp[0]] = tData['model'][temp[1]].permute((1,0)).cpu().detach().numpy()
#     else:
#         paddle_dict[temp[0]] = tData['model'][temp[1]].cpu().detach().numpy()

# for temp in paddle_dict.keys():
#     print(paddle_dict[temp].shape, pData[temp].shape)

# paddle.save(paddle_dict, 'tresnet_m.pdparams')
