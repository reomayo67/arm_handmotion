import sys
import os
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import ViTForImageClassification, ViTImageProcessor
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt6.QtCore import Qt, QPoint

# Setup PyTorch Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Downloading Vision Transformer and Image Processor...")
print("(This may take a minute, the ViT is a heavier model)")
processor = ViTImageProcessor.from_pretrained("WinKawaks/SketchXAI-Base-QuickDraw345")
model = ViTForImageClassification.from_pretrained("WinKawaks/SketchXAI-Base-QuickDraw345")
model.to(device)
model.eval()

class QuickDrawApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quick Draw AI - Ultra Accurate ViT")
        
        # Setup PyQt Canvas
        self.canvas_size = 512
        self.image = QImage(self.canvas_size, self.canvas_size, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.last_point = QPoint()
        self.drawing = False

        # Setup UI Layout
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        self.canvas_label = QLabel()
        self.update_canvas()
        layout.addWidget(self.canvas_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.result_label = QLabel("ViT Model loaded. Draw something!")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.result_label.font()
        font.setPointSize(12)
        self.result_label.setFont(font)
        layout.addWidget(self.result_label)
        
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Clear Canvas")
        clear_btn.clicked.connect(self.clear_canvas)
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.predict)
        
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(predict_btn)
        layout.addLayout(btn_layout)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def update_canvas(self):
        self.canvas_label.setPixmap(QPixmap.fromImage(self.image))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = self.canvas_label.mapFrom(self, event.pos())

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            current_point = self.canvas_label.mapFrom(self, event.pos())
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.GlobalColor.black, 14, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update_canvas()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def clear_canvas(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update_canvas()
        self.result_label.setText("Canvas cleared.")

    def predict(self):
        temp_path = "temp_canvas.png"
        self.image.save(temp_path)
        
        # Load image as standard RGB
        pil_img = Image.open(temp_path).convert("RGB")
        
        # Invert to black background / white strokes (matching Quick Draw dataset format)
        inverted = ImageOps.invert(pil_img)
        
        # Check if the canvas is empty using a grayscale conversion for the bounding box
        grayscale = inverted.convert("L")
        bbox = grayscale.getbbox()
        if not bbox:
            self.result_label.setText("Canvas is empty!")
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return
            
        # Crop to the drawing and add padding
        cropped = inverted.crop(bbox)
        padded = ImageOps.expand(cropped, border=40, fill=(0, 0, 0))
        
        # The ViT processor automatically handles the complex resizing (to 224x224), 
        # normalization, and mapping to PyTorch tensors
        inputs = processor(images=padded, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            
        probs_np = probabilities.cpu().numpy()
        top_3_indices = np.argsort(probs_np)[-3:][::-1]
        
        result_text = "Top Predictions (Vision Transformer):\n"
        for i in top_3_indices:
            confidence = probs_np[i] * 100
            # The ViT model config comes pre-packaged with the string labels
            label = model.config.id2label[i]
            result_text += f"{label.title()}: {confidence:.1f}%\n"
            
        self.result_label.setText(result_text)
        
        # Clean up
        try:
            os.remove(temp_path)
        except OSError:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuickDrawApp()
    window.show()
    sys.exit(app.exec())