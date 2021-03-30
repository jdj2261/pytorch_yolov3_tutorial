from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable     # autograd : 자동미분기능 
import numpy as np
from collections import OrderedDict

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
    blocks.append(dict(block))

    return blocks
print(parse_cfg("cfg/yolov3.cfg"))

# print(len(parse_cfg("cfg/yolov3.cfg")))