# gui/image_window.py
import tkinter as tk
from tkinter import filedialog
import os  # ADDED: Critical import for os.path
from PIL import Image, ImageTk
from .prediction import predict_digit

class ImageRecognitionWindow:
    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        
        # Create new window
        self.window = tk.Toplevel(parent)
        self.window.title("Image Recognition")
        self.window.geometry("350x480")
        self.window.resizable(False, False)
        
        # --- Image Display Area ---
        self.image_display = tk.Canvas(
            self.window, 
            width=280, 
            height=280, 
            bg="white"
        )
        self.image_display.pack(pady=(15, 5))
        
        # --- Image Buttons ---
        self.create_buttons()
        
        # --- Result display ---
        self.create_result_display()
        
        # --- Status label ---
        self.status_label = tk.Label(
            self.window,
            text="Load an image to recognize digits",
            font=("Arial", 10),
            fg="#616161"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Initialize image variables
        self.current_image = None
        self.photo_image = None
    
    def create_buttons(self):
        img_button_frame = tk.Frame(self.window)
        img_button_frame.pack(pady=(5, 10))
        
        tk.Button(
            img_button_frame, 
            text="Load Image", 
            command=self.load_image, 
            width=10, 
            bg="#2196F3", 
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            img_button_frame, 
            text="Predict", 
            command=self.predict, 
            width=10, 
            bg="#4CAF50", 
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            img_button_frame, 
            text="Close", 
            command=self.window.destroy, 
            width=10, 
            bg="#9E9E9E", 
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
    
    def create_result_display(self):
        self.result_frame = tk.Frame(self.window, height=100)
        self.result_frame.pack(pady=(10, 5), fill="x")
        self.result_frame.pack_propagate(False)
        
        self.result_label = tk.Label(
            self.result_frame, 
            text="?", 
            font=("Arial", 64, "bold"),
            width=2,
            height=1
        )
        self.result_label.pack(pady=(5, 0))
        
        self.confidence_label = tk.Label(
            self.result_frame, 
            text="Confidence: N/A", 
            font=("Arial", 14)
        )
        self.confidence_label.pack(pady=(0, 5))
    
    def load_image(self):
        """Load image into the image window"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            # Load and resize for display
            img = Image.open(file_path).convert('L')
            
            # Resize for display (keep aspect ratio)
            display_size = (280, 280)
            w, h = img.size
            scale = min(display_size[0]/w, display_size[1]/h)
            new_size = (int(w*scale), int(h*scale))
            img_display = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(img_display)
            
            # Update display
            self.image_display.delete("all")
            x = (280 - new_size[0]) // 2
            y = (280 - new_size[1]) // 2
            self.image_display.create_image(x, y, anchor=tk.NW, image=self.photo_image)
            
            # Store original for processing
            self.current_image = img
            
            self.status_label.config(
                text=f"Loaded: {os.path.basename(file_path)}",
                fg="#1976D2"
            )
            
            # Auto-predict
            self.predict()
            
        except Exception as e:
            self.status_label.config(
                text=f"Error: {str(e)}",
                fg="#D32F2F"
            )
    
    def predict(self):
        """Predict digit from loaded image"""
        if self.current_image is None:
            self.status_label.config(
                text="Load an image first",
                fg="#D32F2F"
            )
            return
        
        from .model_loader import load_digit_recognition_model
        model = load_digit_recognition_model()
        
        if not model:
            self.status_label.config(
                text="Model not loaded",
                fg="#D32F2F"
            )
            return
        
        # Use shared prediction logic
        result = predict_digit(model, self.current_image)
        
        if result["success"]:
            # Display result with proper spacing
            self.result_label.config(text=str(result["digit"]))
            
            # Color code based on confidence
            if result["confidence"] > 0.9:
                self.result_label.config(fg="#388E3C")
                self.confidence_label.config(fg="#388E3C")
                self.status_label.config(
                    text="High confidence prediction", 
                    fg="#388E3C"
                )
            elif result["confidence"] > 0.7:
                self.result_label.config(fg="#F57C00")
                self.confidence_label.config(fg="#F57C00")
                self.status_label.config(
                    text="Medium confidence prediction", 
                    fg="#F57C00"
                )
            else:
                self.result_label.config(fg="#D32F2F")
                self.confidence_label.config(fg="#D32F2F")
                self.status_label.config(
                    text="Low confidence - try another image", 
                    fg="#D32F2F"
                )
            
            self.confidence_label.config(text=f"Confidence: {result['confidence']:.2%}")
            self.status_label.config(
                text=f"Done in {result['time']:.2f}s | {result['confidence']:.0%} conf",
                fg="#388E3C" if result["confidence"] > 0.7 else "#D32F2F"
            )
        else:
            self.status_label.config(
                text=f"Error: {result['error']}",
                fg="#D32F2F"
            )