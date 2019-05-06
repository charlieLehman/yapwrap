from torchvision.datasets import CocoDetection
from pycocotools import mask
from pycocotools.coco import COCO
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from tqdm import tqdm
import os
from PIL import Image

cc = COCO('data/MSCOCO/annotations/instances_train2017.json')
cats = {c['id']:c['name'] for c in cc.loadCats(cc.getCatIds())}
print(cats)

for split in ['train', 'val']:
    coco = CocoDetection('data/MSCOCO/images/{}2017'.format(split), 'data/MSCOCO/annotations/instances_{}2017.json'.format(split))

    dest = 'data/MSCOCO/imageclassification/{}/'.format(split)

    if not os.path.exists(dest):
        os.mkdir(dest)

    ii = 0
    for x, y in tqdm(coco):
        w,h = x.size
        x = np.array(x)
        for _y in y:
            cat = _y['category_id']
            rle = mask.frPyObjects(_y['segmentation'], h,w)
            mm = mask.toBbox(rle)
            for m in mm:
                if m.shape==(4,):
                    m = [int(u) for u in m]
                    x1,x2,y1,y2 = m[0],m[0]+m[2], m[1],m[1]+m[3]
                    if m[2] > 32 and m[3] > 32:
                        im = Image.fromarray(x[y1:y2,x1:x2])
                        la = '{:03d}_{}'.format(_y['category_id'], cats[_y['category_id']])
                        la_path = os.path.join(dest,la)
                        if not os.path.exists(la_path):
                            os.mkdir(la_path)
                        im_path = os.path.join(la_path,'{:08d}.jpg'.format(ii))
                        im.save(im_path)
                        ii += 1


