import os
import sys
from pathlib import Path
from tkinter.messagebox import NO
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import classify_transforms
from yolov5.utils.dataloaders import LoadImage, LoadImages
from yolov5.utils.general import (LOGGER, Profile, check_img_size, cv2,
                           non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, smart_inference_mode

class Yolov5:
    @smart_inference_mode()
    def __init__(self, weights=ROOT / 'yolov5x6.pt',  # model.pt path(s)
                dnn=False,  # use OpenCV DNN for ONNX inference
                data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                half=False,  # use FP16 half-precision inference
                imgsz=(1280, 1280),  # inference size (height, width)
                device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                ):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.model.pt else 1, 3, *self.imgsz))  # warmup
        

    @smart_inference_mode()
    def inference(self, img_folder, save_folder,
                    txt_folder, # save results to *.txt
                    save_conf=True,  # save confidences in --save-txt labels
                    augment=False,  # augmented inference,
                    visualize=False,  # visualize features,
                    conf_thres=0.25,  # confidence threshold
                    iou_thres=0.45,  # NMS IOU threshold
                    max_det=1000,  # maximum detections per image,
                    classes=None,  # filter by class: --class 0, or --class 0 2 3
                    agnostic_nms=False,  # class-agnostic NMS,
                    line_thickness=1,  # bounding box thickness (pixels)
                    hide_labels=False,  # hide labels
                    hide_conf=False,  # hide confidences
                ):
        # Dataloader
        dataset = LoadImages(img_folder, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt)
        bs = 1  # batch_size

        # Run inference
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            if im is None:
                continue
            seen += 1
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                pred = self.model(im, augment=augment, visualize=visualize)
            # NMS
            pred = torch.unsqueeze(pred, 0)
            # print(pred)
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = os.path.join(save_folder , p.name)  # im.jpg
                txt_path = os.path.join(txt_folder , p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.model.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if txt_path:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_folder:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (self.model.names[c] if hide_conf else f'{self.model.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                # Stream results
                im0 = annotator.result()
                # Save results (image with detections)
                if save_folder:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
            # Print time (inference-only)
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        info = str(f'%.1f\t%.1f\t%.1f\t{(1, 3, *self.imgsz)}\t' % t)+ str(seen)
        return info

if __name__ == "__main__":
    model_path = Yolov5()
    text = model_path.inference('/home/trandat/project/p51/object_detection/backend/data_test/src/1.jpg', '/home/trandat/project/p51/object_detection/backend/data_test/result/1.jpg', '/home/trandat/project/p51/object_detection/backend/data_test/result/1.txt')