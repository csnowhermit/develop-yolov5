import torch
import torch.nn as nn

# 测试BCELoss和BCEWithLogitsLoss的区别
label = torch.Tensor([1, 1, 0])
pred = torch.Tensor([3, 2, 1])
pred_sig = torch.sigmoid(pred)

loss = nn.BCELoss()
print(loss(pred_sig, label))

loss = nn.BCEWithLogitsLoss()
print(loss(pred, label))

loss = nn.BCEWithLogitsLoss()
print(loss(pred_sig, label))
