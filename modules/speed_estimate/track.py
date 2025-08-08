import sys
sys.path.insert(0, './yolov5')
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import copy
import math
from modules.speed_estimate.yolov5.utils.google_utils import attempt_download
from modules.speed_estimate.yolov5.models.experimental import attempt_load
from modules.speed_estimate.yolov5.utils.datasets import LoadImages, LoadStreams
from modules.speed_estimate.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
import pandas as pd
from modules.speed_estimate.yolov5.utils.torch_utils import select_device, time_sync
from modules.speed_estimate.deep_sort_pytorch.utils.parser import get_config
from modules.speed_estimate.deep_sort_pytorch.deep_sort import DeepSort
global np_store,exclamation
np_store = np.zeros((500, 6))
alpha = 1.8/1.3
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def three_dimension(bboxx, bboxy, focallength=1507, principlepoint=(986.4,536.8), H = 6.29,theta=3.14*70/180):
    # bboxcorner : real pixel position of bboxcorner which is near to ground
    # imsize : image.shape[:2]
    # focal length : focal length of the camera (from matlab calibration application)
    # principlepoint : (px, py) (from matlab calibration application)

    if bboxx == 0 and bboxy == 0:
        return [0,0,0]
    (px,py) = principlepoint

    # H : measured height of the camera with respect to ground
    # theta : measured angle of the camera

    Z = H * math.tan(theta + math.atan((py - bboxy)/focallength))

    X = (bboxx-px) * (Z/math.cos(theta)) / focallength
    Y = 0

    return np.array([X,Y,Z])


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, frame_idx, myclasses, identities=None, offset=(0, 0), results_list=None):
    names = ['no classification','car', 'mini bus', 'large bus', 'truck', 'large trailer', 'motorcycle', 'pedestrian']
    global np_store
    how_many_frame_will_detect = 30
    video_frame_rate = 30
    for i, box in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        classes = int(myclasses[i]) if myclasses is not None else 0
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cen_x = int((x1 + x2) / 2)
        cen_y = int((y1 + y2) / 2)

        if i > np_store.shape[0]:
            np_store = np.vstack((np_store, np.zeros((500, 6))))
        if np_store[i][0] == 0:
            np_store[i][0] = frame_idx
            np_store[i][1] = id

        if (frame_idx >= how_many_frame_will_detect + 1) and (frame_idx % how_many_frame_will_detect == 0):
            np_store[i][2] = copy.deepcopy(np_store[i][4])
            np_store[i][3] = copy.deepcopy(np_store[i][5])
            np_store[i][4] = cen_x
            np_store[i][5] = cen_y

        loc1 = three_dimension(np_store[i][2], np_store[i][3])
        loc2 = three_dimension(cen_x, cen_y)

        if loc2[0] == 0 or loc1[0] == 0:
            velocity_y = 0
        else:
            velocity_y = alpha * math.sqrt(sum((loc2 - loc1) ** 2)) / ((how_many_frame_will_detect) / video_frame_rate)
        try:
            km_hr_v = 3.6 * velocity_y
            class_names = names[classes]
            color = compute_color_for_labels(id)

            label = '{}{} {}Km/hr, id:{}'.format("", class_names, int(km_hr_v), id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.circle(img, (cen_x, cen_y), 3, color, -1)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

            if results_list is not None:
                results_list.append({
                    'frame': frame_idx,
                    'id': id,
                    'class': class_names,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'speed_kmh': round(km_hr_v, 2)
                })
        except:
            continue
    return img

def run_speed_estimate(
        source,
        yolo_weights='modules/speed_estimate/best_yolov5s_aihub_finetune.pt',
        deep_sort_weights='modules/speed_estimate/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
        output='inference/output',
        img_size=640,
        conf_thres=0.4,
        iou_thres=0.5,
        fourcc='mp4v',
        device='',
        show_vid=False,
        save_vid=False,
        save_txt=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        evaluate=True,
        config_deepsort='modules/speed_estimate/deep_sort_pytorch/configs/deep_sort.yaml'
    ):
    # 判断是不是摄像头输入
    webcam = source == '0' or source == 0

    # 初始化 DeepSort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    device = select_device(device)
    half = device.type != 'cpu'

    if not evaluate:
        if os.path.exists(output):
            shutil.rmtree(output)
        os.makedirs(output)

    model = attempt_load(yolo_weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()

    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    video_name = os.path.splitext(os.path.basename(source))[0]
    output_csv_path = os.path.join(".", "analysis_results", video_name, "[V2]vehicle_speed.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    csv_data = []

    vid_path, vid_writer = None, None
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], f'{i}: ', im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            save_path = str(Path(output) / Path(p).name)
            s += f'{img.shape[2]}x{img.shape[3]} '

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                xywh_bboxs = []
                confs = []
                myclasses = det[:, -1]

                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                outputs = deepsort.update(xywhs, confss, myclasses, im0)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    myclassesyo = outputs[:, -1]
                    identities = outputs[:, 4]
                    draw_boxes(im0, bbox_xyxy, frame_idx, myclassesyo, identities, results_list=csv_data)

                    if save_txt:
                        txt_path = os.path.join(output, f"{video_name}.txt")
                        for j, (tlwh_bbox, output) in enumerate(zip(xyxy_to_tlwh(bbox_xyxy), outputs)):
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, output[4], tlwh_bbox[0], tlwh_bbox[1],
                                                              tlwh_bbox[2], tlwh_bbox[3], output[5], -1, -1, -1))
            else:
                deepsort.increment_ages()

            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration

            if save_vid:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.path.abspath(output))
    print('Done. (%.3fs)' % (time.time() - t0))

    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Per-frame speed results saved to {output_csv_path}")

