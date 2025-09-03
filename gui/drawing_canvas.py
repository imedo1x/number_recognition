# gui/drawing_canvas.py
import tkinter as tk
from PIL import Image, ImageDraw
from .prediction import predict_digit

class DrawingCanvas:
    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        
        # --- Canvas for drawing ---
        self.canvas = tk.Canvas(parent, width=280, height=280, bg="black", cursor="cross")
        self.canvas.pack(pady=(15, 5))
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_draw)
        
        # --- Create drawing image ---
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.drawing = False
        
        # --- Main Buttons ---
        self.create_buttons()
        
        # --- Result display ---
        self.create_result_display()
    
    def create_buttons(self):
        button_frame = tk.Frame(self.parent)
        button_frame.pack(pady=(5, 10))
        
        tk.Button(
            button_frame, 
            text="Predict", 
            command=self.predict, 
            width=10, 
            bg="#4CAF50", 
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, 
            text="Clear", 
            command=self.clear, 
            width=10, 
            bg="#F44336", 
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, 
            text="Photos", 
            command=self.app.open_image_window, 
            width=10, 
            bg="#2196F3", 
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
    
    def create_result_display(self):
        # --- Result display ---
        self.result_frame = tk.Frame(self.parent, height=100)
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
    
    def start_draw(self, event):
        """Start drawing on canvas"""
        self.last_x, self.last_y = event.x, event.y
        self.drawing = True
    
    def paint(self, event):
        """Draw on canvas with smooth lines"""
        if self.last_x and self.last_y and self.drawing:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=15, fill='white', 
                capstyle=tk.ROUND,
                smooth=tk.TRUE
            )
            
            # Draw on image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y], 
                fill='white', 
                width=15
            )
            
            self.last_x, self.last_y = event.x, event.y
        elif self.drawing:
            self.last_x, self.last_y = event.x, event.y
    
    def clear(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.drawing = False
        self.result_label.config(text="?")
        self.confidence_label.config(text="Confidence: N/A")
        self.app.status_bar.set("Canvas cleared - draw a new digit")
    
    def predict(self):
        """Predict hand-drawn digit"""
        from .model_loader import load_digit_recognition_model
        
        model = load_digit_recognition_model()
        if not model:
            self.app.status_bar.set("Error: Model failed to load", "error")
            return
        
        # Use shared prediction logic
        result = predict_digit(model, self.image)
        
        if result["success"]:
            # Display result with proper spacing
            self.result_label.config(text=str(result["digit"]))
            
            # Color code based on confidence
            if result["confidence"] > 0.9:
                self.result_label.config(fg="#388E3C")
                self.confidence_label.config(fg="#388E3C")
                self.app.status_bar.set("High confidence prediction", "success")
            elif result["confidence"] > 0.7:
                self.result_label.config(fg="#F57C00")
                self.confidence_label.config(fg="#F57C00")
                self.app.status_bar.set("Medium confidence prediction", "warning")
            else:
                self.result_label.config(fg="#D32F2F")
                self.confidence_label.config(fg="#D32F2F")
                self.app.status_bar.set("Low confidence - try redrawing", "error")
            
            self.confidence_label.config(text=f"Confidence: {result['confidence']:.2%}")
        else:
            self.app.status_bar.set(f"Prediction error: {result['error']}", "error")