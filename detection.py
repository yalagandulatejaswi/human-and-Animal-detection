import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path=None, num_classes=91):
        """Initialize Faster R-CNN detector"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load pretrained Faster R-CNN with updated weights API
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # COCO class names (person=1, various animals)
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect(self, frame, conf_threshold=0.3):
        """Detect objects in frame - returns ALL objects above threshold"""
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        # Keep ALL detections above threshold (not just specific classes)
        keep = predictions['scores'] > conf_threshold
        
        return {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy()
        }
    
    def crop_detections(self, frame, results):
        """Crop detected objects from frame"""
        crops = []
        
        for i, box in enumerate(results['boxes']):
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure valid crop coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                label_idx = int(results['labels'][i])
                crops.append({
                    'image': crop,
                    'box': (x1, y1, x2, y2),
                    'conf': float(results['scores'][i]),
                    'cls': label_idx,
                    'class_name': self.coco_names[label_idx] if label_idx < len(self.coco_names) else 'unknown'
                })
        
        return crops
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.005):
        """Train detection model"""
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        return self.model
