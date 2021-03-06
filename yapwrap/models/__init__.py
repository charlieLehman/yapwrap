from .resnet import *
from .complementconstraint import *
from .implicit_attention import *
from .tinypreactresnetattn import *
from .tinywideresnetattn import *
from .convnet import *
from .alexnet import *
from .segnet import *
from .attention import *
from .dssnet import DSSNet
from .poolnet import PoolNetResNet50, TinyPoolNetResnet18
from .deeplab_imclass import *

def ComplementConstraintTinyAttention18(num_classes=10):
    return ComplementConstraint(TinyAttention18(num_classes=num_classes))
