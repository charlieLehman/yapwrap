from .tinyresnet import *
from .complementconstraint import *
from .tinyresnetattn import *
from .tinypreactresnetattn import *
from .convnet import *
from .alexnet import *


def ComplementConstraintTinyAttention18(num_classes=10):
    return ComplementConstraint(TinyAttention18(num_classes=num_classes))
