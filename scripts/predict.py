# scripts/predict.py
import torch
import numpy as np
from PIL import Image
import sys
import os

def predict_digit(image_path):
    """Predict digit from image file"""
    # --- Model Architecture ---
    class ImprovedCNN(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, 1, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, 1, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25),
                torch.nn.Conv2d(32, 64, 3, 1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, 1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25),
                torch.nn.Conv2d(64, 128, 3, 1, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25)
            )
            
            # Calculate flattened size
            dummy_input = torch.zeros(1, 1, 28, 28)
            self.flattened_size = self.features(dummy_input).view(1, -1).shape[1]
            
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.flattened_size, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "digit_cnn_improved.pth")
    
    # --- Load Model ---
    model = ImprovedCNN(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # --- Preprocess Image ---
    def preprocess_image(image_path):
        # Load image
        image = Image.open(image_path).convert('L')  # Grayscale
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy and check if background is dark
        img_array = np.array(image)
        if img_array.mean() < 128:  # Dark background
            img_array = 255 - img_array  # Invert to white-on-black
        
        # Convert back to PIL
        image = Image.fromarray(img_array)
        
        # Transform
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        return transform(image).unsqueeze(0)  # Add batch dimension
    
    # --- Predict ---
    try:
        tensor = preprocess_image(image_path)
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), confidence.item()
    
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("🧠 Digit Recognition Predictor")
    print("="*30)
    
    while True:
        image_path = input("\nEnter path to digit image (or 'quit' to exit): ").strip()
        if image_path.lower() == 'quit':
            break
            
        if not os.path.exists(image_path):
            print("❌ File not found! Please check the path.")
            continue
            
        digit, confidence = predict_digit(image_path)
        if digit is not None:
            print(f"🎯 PREDICTION: {digit}")
            print(f"📊 CONFIDENCE: {confidence:.2%}")
            
            # Show warning for low confidence
            if confidence < 0.85:
                print("⚠️ WARNING: Low confidence - result may be unreliable")