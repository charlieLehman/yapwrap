from yapwrap.dataloaders import *
from tqdm import tqdm
# MSRAB
# trainset = MSRABDataset(root='/data/datasets/MSRA-B', split='train', transform=yapwrap_transforms.ToTensor())
# mean = torch.zeros(3)
# std = torch.zeros(3)
# total = len(trainset)
# for im,la in tqdm(trainset):
#     mean += im.mean((-1,-2))/255
#     std += im.std((-1,-2))/255
# print(mean/total)
# print(std/total)

# LID

d = LIDTrack1('/data/datasets/LID', 224, batch_sizes={'train':128,'val':20,'test':20})

from tqdm import tqdm
for x, l in tqdm(d.train_iter()):
    print(x.shape, l.shape)

