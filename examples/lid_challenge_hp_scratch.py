from yapwrap.dataloaders import *
import torchvision.transforms as tvtfs
import yapwrap.dataloaders.yapwrap_transforms as tfs
from tqdm import tqdm
from matplotlib import pyplot as plt

train_transform = tvtfs.Compose([
# tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
tfs.ToTensor()])

hp = HyperpolarLabelDataset('data/LID_training_msrab', train_transform, 200)

for image, saliency, class_label in hp:

    fig, ax = plt.subplots(2)

    ax[0].imshow(image.permute([1,2,0])/255)
    ax[1].imshow(saliency.squeeze())
    plt.title(class_label)
    plt.show()
