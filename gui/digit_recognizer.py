# gui/digit_recognizer.py
import tkinter as tk
from tkinter import Canvas, Button, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import torch
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("400x500")
        
        # --- Canvas for drawing ---
        self.canvas = Canvas(root, width=280, height=280, bg="black")
        self.canvas.pack(pady=20)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # --- Create drawing image ---
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # --- Buttons ---
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        self.predict_btn = Button(button_frame, text="Predict", command=self.predict, width=10, bg="#4CAF50", fg="white")
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = Button(button_frame, text="Clear", command=self.clear_canvas, width=10, bg="#F44336", fg="white")
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_btn = Button(button_frame, text="Load Image", command=self.load_image, width=10, bg="#2196F3", fg="white")
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # --- Result display ---
        self.result_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
        self.result_frame.pack(fill=tk.X, padx=20)
        
        tk.Label(self.result_frame, text="Prediction:", font=("Arial", 14)).pack()
        self.result_label = tk.Label(self.result_frame, text="?", font=("Arial", 48, "bold"), fg="#2196F3")
        self.result_label.pack()
        
        self.confidence_label = tk.Label(self.result_frame, text="Confidence: N/A", font=("Arial", 12))
        self.confidence_label.pack()
        
        # --- Load model ---
        self.model = self.load_model()
        if self.model is None:
            messagebox.showerror("Error", "Failed to load model. Please check if model file exists.")
            self.root.destroy()
            return
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Model architecture must match training
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
            
            # Load model
            model_path = os.path.join(PROJECT_ROOT, "models", "digit_cnn_improved.pth")
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return None
                
            model = ImprovedCNN(num_classes=10)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            print("✅ Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return None
    
    def paint(self, event):
        """Draw on canvas"""
        r = 8
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='white', outline='white')
        self.draw.ellipse([event.x-r, event.y-r, event.x+r, event.y+r], fill='white')
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="?")
        self.confidence_label.config(text="Confidence: N/A")
    
    def load_image(self):
        """Load an image from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if file_path:
            try:
                # Load and resize image
                img = Image.open(file_path).convert('L')
                img = img.resize((280, 280), Image.Resampling.LANCZOS)
                
                # Check if background is dark
                if np.array(img).mean() < 128:
                    img = ImageOps.invert(img)
                
                # Update canvas
                self.clear_canvas()
                self.image = img.copy()
                self.draw = ImageDraw.Draw(self.image)
                
                # Display on canvas
                photo = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.image = photo  # Keep reference
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def predict(self):
        """Predict the drawn digit"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        try:
            # Preprocess image
            img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            
            # Normalize and prepare for model
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            tensor = transform(img_array).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                output = self.model(tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Display result
            digit = predicted.item()
            confidence_val = confidence.item()
            
            self.result_label.config(text=str(digit))
            
            # Color code based on confidence
            if confidence_val > 0.9:
                self.result_label.config(fg="#4CAF50")  # Green
            elif confidence_val > 0.7:
                self.result_label.config(fg="#FFC107")  # Amber
            else:
                self.result_label.config(fg="#F44336")  # Red
            
            self.confidence_label.config(text=f"Confidence: {confidence_val:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()