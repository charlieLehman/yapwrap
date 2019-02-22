import os

class Dataloader(object):
    def __init__(self, root, size, batch_sizes, transforms):

        self.name = self.__class__.__name__

        if root is not None:
            if not isinstance(root, str):
                raise TypeError('{} is not a valid type str'.format(type(root).__name__))
            if not os.path.exists(root):
                print('creating {}'.format(root))
                os.mkdir(root)
        self.root = root
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise TypeError('{} is not a valid type (int,int) or int'.format(type(size).__name__))
        if not isinstance(batch_sizes, dict):
            raise TypeError('{} is not a valid type with format {\'train\':int,\'test\':int, \'val\':int,...etc.}'.format(type(batch_sizes).__name__))

        if not isinstance(transforms, dict):
            raise TypeError('{} is not a valid type with format {\'train\':transform,\'test\':transform, \'val\':transform,...etc.}'.format(type(transforms).__name__))

        for v in transforms.values():
            if not callable(v) and v is not None:
                raise TypeError('{} is not a valid transform'.format(type(v).__name__))
        str_transforms = {}
        for i,v in transforms.items():
            try:
                vname = v.__class__.__name__
            except:
                vname = v.__name__
            str_transforms.update({i:vname})

        self.param_dict = {'root':root,
                      'size':size,
                      'batch_sizes':batch_sizes,
                      'transforms':str_transforms}

    def train_iter(self):
        raise NotImplementedError

    def val_iter(self):
        raise NotImplementedError

    def test_iter(self):
        raise NotImplementedError

    def ood_iter(self):
        raise NotImplementedError

    @property
    def class_names(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def examples(self):
        raise NotImplementedError
    @property
    def params(self):
        return self.param_dict
    def __repr__(self):
        return str(self.params)

