# gui/prediction.py
import numpy as np
import cv2
import time
import torch  # CRITICAL FIX: Added missing torch import
import os

def preprocess_image(img, is_drawing=False):
    """Preprocess image for prediction"""
    try:
        img_np = np.array(img)
        
        # AUTO INVERSION (for real-world images)
        if not is_drawing and img_np.mean() > 128:  # Light background
            img_np = 255 - img_np
        
        # Resize to 28x28 (preserving aspect ratio)
        h, w = img_np.shape
        if h > w:
            new_h = 20
            new_w = int(w * 20 / h)
            pad_w = (28 - new_w) // 2
            pad_h = 4
        else:
            new_w = 20
            new_h = int(h * 20 / w)
            pad_h = (28 - new_h) // 2
            pad_w = 4
        
        # Resize and pad
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        processed = np.pad(
            resized, 
            ((pad_h, pad_h), (pad_w, pad_w)), 
            mode='constant', 
            constant_values=0
        )
        
        return processed
        
    except Exception as e:
        return None

def predict_digit(model, image):
    """Predict digit from image (works for both drawing and photos)"""
    try:
        start_time = time.time()
        
        # Determine if this is a drawing or photo
        is_drawing = (image.size == (280, 280))
        
        # Preprocess image
        processed = preprocess_image(image, is_drawing)
        if processed is None:
            return {
                "success": False,
                "error": "Image preprocessing failed"
            }
        
        # Transform for model
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor = transform(processed).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            "success": True,
            "digit": predicted.item(),
            "confidence": confidence.item(),
            "time": time.time() - start_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }