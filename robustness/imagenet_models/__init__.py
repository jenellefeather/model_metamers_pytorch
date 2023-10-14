from .resnet import *
from .vgg import *
from .alexnet import *
from .alexnet_reduced_aliasing import * 
from .cornet_s import * 
from .hmax import * 
from .resnet_openselfsup_transfer import * 
from .vonenet import * 

# Shape trained models
from .texture_shape_models import *

# For the ipcl models
from .ipcl_alexnet_gn import ipcl_alexnet_gn

# For VITs, CLIP, and SWSL
from .vision_transformer import *
from .timm_resnet_gelu import resnet50_gelu, swsl_resnext101_32x8d
