# util.py
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def unique(tensor):
    """
    동일한 클래스에 다수의 true ditections가 있을 수 있으므로,
    중복되지 않은 class들을 가져온다.
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    detection feature map을 받아서 2-D tensors로 변환을 한다.
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguos()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # x,y 좌표와 objectness 점수 sigmoid
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    grid = np.arange(grid_size)     # 벡터로 변환
    a, b = np.meshgrid(grid, grid)  # 벡터를 행렬로 변환

    x_offset = torch.FloatTensor(a).view(-1,1)  # tensor로 변환
    y_offset = torch.FloatTensor(b).view(-1,1)  # tensor로 변환

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset),1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    # Bounding box 크기에 anchors 적용
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0) # 0은 첫번째 차원
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # class 점수에 sigmoid activation 적용
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    # 입력 이미지 크기에 맞게 resize
    # feature map size : 13 * 13
    # 입력 이미지가 416 * 416이면, 속성들에 stride variable을 곱해준다.
    prediction[:,:,:4] *= stride
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    input : prediction, confidence(objectness score threshold), num_classes(80, COCO)
            nms_conf (NMS IoU threshold)
    """
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # 기존 속성(중심 좌표 x, y, 높이, 너비)을 
    # (왼쪽 위 꼭지점 x, y, 오른쪽 밑 꼭지점 x,y)로 변환한다.
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    """
    True detection 수는 이미지마다 다르므로
    confidence thresholding과 NMS가 이미지당 한번씩 처리되어야 한다.
    벡터화시키는 것이 불가능하므로 prediction의 첫 번째 dimension을 loop해야 한다.
    """
    batch_size = prediction.size(0)

    write = False
    for index in range(batch_size):
        image_pred = prediction[index]
        #confidence threshholding 
        #NMS
    
        # 가장 높은 값을 가진 class score를 제외하고 모두 삭제
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        # threshold보다 낮은 object confidence를 지닌 bounding box rows를 0으로 설정한 것을 제거
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        # PyToch 0.4 호환성
        # scalar가 PyTorch 0.4에서 지원되기 때문에 no detection에 대한 
        # not raise exception 코드입니다.       
        if image_pred_.shape[0] == 0:
            continue       

        # 이미지에서 검출된 다양한 classes를 얻기
        img_classes = unique(image_pred_[:,-1]) # -1 index는 class index를 지니고 있습니다.

        for cls in img_classes:
            # NMS 실행
            # 특정 클래스에 대한 detections 얻기
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # 가장 높은 objectness를 지닌 detections 순으로 정렬하기
            # confidence는 가장 위에 있다.
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #detections의 수
        
            for i in range(idx):
                # 모든 박스에 대해 하나하나 IoU 얻기
                try:
                    # i 인덱스를 갖고 있는 box의 IoU와 i보다 큰 인덱스를 지닌 bounding boxes 얻기
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # IoU > threshold인 detections를 0으로 만들기
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # non-zero 항목 제거하기
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            """
            write_results 함수는 Dx8 크기의 tensor를 출력.
            이미지의 인덱스, 4개의 꼭지점 좌표, objectness score, 
            가장 높은 클래스 score, 그 class의 인덱스
            """
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(index) 
            # 이미지에 있는 class의 detections 만큼 batch_id를 반복합니다.
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output
    except:
        return 0

def letterbox_image(img, inp_dim):
    ''' padding을 사용하여 aspect ratio가 변화하지 않고 이미지를 resize 합니다.'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas
    
def prep_image(img, inp_dim):
    """
    신경망에 입력하기 위한 이미지 준비
    변수를 반환합니다. 
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile: str) -> list:
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names