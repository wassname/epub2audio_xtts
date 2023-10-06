# %%
# check if cuda works
import torch
print(torch.cuda.is_available())

t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
# %%
