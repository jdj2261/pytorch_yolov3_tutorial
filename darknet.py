from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable     # autograd : 자동미분기능 
import numpy as np
from collections import OrderedDict

from torch.nn.modules import module

def parse_cfg(cfgfile: str) -> list:
    """
    cfg 파일을 파싱해서 모든 block을 dict 형식으로 저장
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """

    # cfg파일 내용 string list에 저장
    file = open(cfgfile, 'r')       # cfg 파일 읽기
    lines = file.read().split('\n')  # lint 형태로 lines에 저장
    lines = [x for x in lines if len(x) > 0]    # 비어있는 line 제거
    lines = [x for x in lines if x[0] != '#']   # 주석들 제거
    lines = [x.strip() for x in lines]          # whitespaces 제거

    block = OrderedDict()
    blocks = []

    for line in lines:
        if line[0] == "[": # 새로운 block의 시작
            if len(block) != 0:
                blocks.append(block) # block list에 추가
                block = {}           # block 비움
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    """
    route는 앞전의 layer를 가져오거나 이어붙이는 일을 한다.
    파이토치에서 torch.cat이라는 코드로 구현할 수 있으나
    이를 구현하는 모듈을 따로 구현하게 되면 과도한 추상화를 야기할 수 있으므로
    Empty Layer로 놔둔 후 forward function에서 처리한다.
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    """
    bounding box들 저장
    """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

# create pytorch module
def create_modules(blocks: list) -> str:
    """
    yolov3.cfg에서 알 수 있듯이 YOLO에는 5개의 Layer가 사용된다.
    이 layer들을 module로 옮기는 함수이다.
    pytorch는 pre-built된 convolutional layer와 upsample layer가 존재한다.
    그 외의 layer들은 nn.Module class를 상속받아 직접 구현한다.
    """
    net_info = blocks[0]            # pre-processing(전처리)와 입력에 관한 정보를 저장
    module_list = nn.ModuleList()   # nn.Module object들을 포함하고 있는 일반적인 list와 유사
    prev_filters = 3                # convolution layer의 filter 개수를 알고 있어야 한다. 초기값은 RGB 3채널이므로 3
    output_filters=[]               # 다음 layer로 진행하기 위해선 이전 layer의 output을 알아야 한다.

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()    # nn.Module object들을 list 형태로 집어넣어 차례대로 실행
        """
        block type 확인
        block에 대해 새로운 module 만들기
        module_list에 추가
        """
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # convolutional layer 추가
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias= bias)
            module.add_module("conv_{}".format(index), conv)

            # batch norm layer 추가
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)

            # activation 확인
            # YOLO는 Linear or Leaky "ReLU"
            if activation == "leaky":
                # inplace = True : input 자체를 수정(메모리 usage는 좋아지나 input을 없앰)
                activn = nn.LeakyReLU(0.1, inplace=True) 
                module.add_module("leaky_{}".format(index), activn)
        
        # upsampling layer이면 Bilinear2dUpsampling 사용
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear") # 2배로 보간하여 업샘플링 적용
            module.add_module("upsample_{}".format(index), upsample)
        

        elif (x["type"] == "route"):
            '''
            Route와 Shortcut Layer 
            Route : index에 해당하는 feature map의 value만을 output으로 내보낸다.
            Shortcut Layer : ResNet에서 사용된 것과 유사한 skip connection
                            ouput에 3번째 전 layer를 더해준다.
            '''
            x["layers"] = x["layers"].split(',')
            # layers수가 1개이면 start만 사용
            start = int(x["layers"][0])
            # 1개 이상이면 end도 사용
            try:
                end = int(x["layers"][1])   
            except:
                end = 0
            # Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)

            # output filter에 들어가는 수는 제대로 유지되어야 하므로
            # filters 업데이트
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        elif (x["type"] == "yolo"):
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]    # mask에 씌여있는 anchor만 남기기

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)


class Darknet(nn.Module):
    """
    custom 구조 만들기
    """
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    """
    network forward 만들기
    """
    def forward(self, x, CUDA):
        modules = self.blocks[1:] # net block 제외
        # route와 shortcut layer는 이전 layer의 output이 필요하므로 저장
        outputs = {} # key : index, value : feature maps

        write = 0
        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x) # forward pass
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)  # 두개의 feature map을 이어붙일 때 사용

            elif module_type = "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_] # 이전레이어와 합쳐주면 됨

            

# blocks = parse_cfg('cfg/yolov3.cfg')
# print(create_modules(blocks))
