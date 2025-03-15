import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk, ImageFilter
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import time, math
import sys, os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import preprocess_image, apply_augmentation

class DigitRecognizer:
    def __init__(self, model_path):
        try:
            if not os.path.isabs(model_path):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(project_root, model_path)
            
            print(f"Looking for model at: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Training a new model...")
                self.train_model(model_path)
            
            print(f"Loading model from {model_path}...")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully.")

            self.augmentation = keras.Sequential([
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1)
            ])

            self.window = tk.Tk()
            self.window.title("Digit Recognizer")
            self.window.geometry("800x600")
            
            main_frame = tk.Frame(self.window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            left_frame = tk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            self.canvas = tk.Canvas(left_frame, width=280, height=280, bg='black')
            self.canvas.pack(pady=5)
            self.image = Image.new('L', (280, 280), 0)
            self.draw = ImageDraw.Draw(self.image)
            
            self.setup_buttons(left_frame)
            
            self.prediction_label = tk.Label(left_frame, text="Prediction: ", font=("Arial", 14))
            self.prediction_label.pack(pady=5)
            
            feedback_frame = tk.Frame(left_frame)
            feedback_frame.pack(pady=5)
            tk.Label(feedback_frame, text="Was the prediction correct?").pack(side=tk.LEFT)
            tk.Button(feedback_frame, text="✓ Yes", bg="lightgreen", command=lambda: self.save_feedback(True)).pack(side=tk.LEFT, padx=5)
            tk.Button(feedback_frame, text="✗ No", bg="lightcoral", command=lambda: self.save_feedback(False)).pack(side=tk.LEFT)
            
            right_frame = tk.Frame(main_frame)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            tk.Label(right_frame, text="Augmentation Preview", font=("Arial", 12)).pack(pady=5)
            self.aug_frame = tk.Frame(right_frame)
            self.aug_frame.pack(pady=5)
            self.aug_canvases = []
            for i in range(4):
                canvas = tk.Canvas(self.aug_frame, width=80, height=80, bg='black')
                canvas.grid(row=i//2, column=i%2, padx=5, pady=5)
                self.aug_canvases.append(canvas)
            
            tk.Label(right_frame, text="Drawing Playback", font=("Arial", 12)).pack(pady=5)
            self.playback_canvas = tk.Canvas(right_frame, width=200, height=200, bg='black')
            self.playback_canvas.pack(pady=5)
            
            playback_controls = tk.Frame(right_frame)
            playback_controls.pack(pady=5)
            self.play_button = tk.Button(playback_controls, text="▶ Play", command=self.play_drawing, state=tk.DISABLED)
            self.play_button.pack(side=tk.LEFT, padx=5)
            self.stop_button = tk.Button(playback_controls, text="■ Stop", command=self.stop_playback, state=tk.DISABLED)
            self.stop_button.pack(side=tk.LEFT, padx=5)
            
            self.stroke_history = []
            self.playback_active = False
            self.playback_index = 0
            self.playback_image = None
            self.playback_draw = None
            
            self.canvas.bind('<Button-1>', self.start_stroke)
            self.canvas.bind('<B1-Motion>', self.continue_stroke)
            self.canvas.bind('<ButtonRelease-1>', self.end_stroke)
            
            self.current_prediction = None
            
        except Exception as e:
            print(f"Error initializing application: {e}")
            raise

    def setup_buttons(self, parent):
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Recognize", command=self.predict).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Generate Augmentations", command=self.update_augmentation_preview).pack(side=tk.LEFT, padx=2)

    def start_stroke(self, event):
        self.current_stroke = []
        self.record_point(event)
        
    def continue_stroke(self, event):
        self.record_point(event)
        self.draw_line(event)
        
    def end_stroke(self, event):
        if self.current_stroke:
            self.stroke_history.append(self.current_stroke)
            self.play_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.predict()
            
    def record_point(self, event):
        self.current_stroke.append({
            'x': event.x,
            'y': event.y,
            'time': time.time()
        })

    def draw_line(self, event):
        r = 12
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='white', outline='white')
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

    def play_drawing(self):
        if not self.stroke_history or self.playback_active:
            return
            
        self.playback_active = True
        self.playback_index = 0
        
        self.playback_image = Image.new('L', (200, 200), 0)
        self.playback_draw = ImageDraw.Draw(self.playback_image)
        
        self.playback_canvas.delete("all")
        
        self.window.after(100, self.playback_next_point)
        
    def playback_next_point(self):
        if not self.playback_active:
            return
            
        total_points = sum(len(stroke) for stroke in self.stroke_history)
        if self.playback_index >= total_points:
            self.playback_active = False
            return
            
        current_index = self.playback_index
        for stroke_idx, stroke in enumerate(self.stroke_history):
            if current_index < len(stroke):
                point = stroke[current_index]
                
                x = int(point['x'] * 200 / 280)
                y = int(point['y'] * 200 / 280)
                
                r = 8
                self.playback_canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
                self.playback_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
                
                self.update_playback_display()
                
                self.playback_index += 1
                self.window.after(50, self.playback_next_point)
                return
            current_index -= len(stroke)
            
        self.playback_active = False
        
    def update_playback_display(self):
        if self.playback_image:
            photo = ImageTk.PhotoImage(self.playback_image)
            self.playback_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.playback_canvas.image = photo
        
    def stop_playback(self):
        self.playback_active = False

    def update_augmentation_preview(self):
        if not self.image:
            return
            
        img_array = preprocess_image(self.image, apply_normalization=False)
        
        for i, canvas in enumerate(self.aug_canvases):
            aug_img = apply_augmentation(img_array, self.augmentation)
            
            aug_pil = Image.fromarray((aug_img[0, :, :, 0] * 255).astype(np.uint8))
            aug_pil = aug_pil.resize((80, 80))
            
            photo = ImageTk.PhotoImage(aug_pil)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

    def predict(self):
        try:
            start_time = time.time()
            img_array = preprocess_image(self.image)
            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit] * 100
            inference_time = (time.time() - start_time) * 1000
            
            self.current_prediction = {
                'digit': int(predicted_digit),
                'confidence': float(confidence),
                'time': datetime.now().isoformat()
            }
            
            self.prediction_label.config(
                text=f"Prediction: {predicted_digit} ({confidence:.1f}%, {inference_time:.0f}ms)",
                fg="green" if confidence > 70 else "red"
            )
            
            self.update_augmentation_preview()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.prediction_label.config(text=f"Error: {e}", fg="red")
            self.current_prediction = None

    def save_feedback(self, is_correct):
        if not self.current_prediction:
            messagebox.showinfo("Feedback", "Make a prediction first!")
            return
            
        feedback = {
            'prediction': self.current_prediction,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            feedback_dir = os.path.join('data', 'feedback')
            os.makedirs(feedback_dir, exist_ok=True)
            
            img_filename = f"digit_{self.current_prediction['digit']}_{int(time.time())}.png"
            img_path = os.path.join(feedback_dir, img_filename)
            self.image.save(img_path)
            
            feedback['image_path'] = img_path
            feedback_file = os.path.join(feedback_dir, 'feedback_log.jsonl')
            
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(feedback) + '\n')
                
            messagebox.showinfo("Feedback", "Thank you for your feedback!")
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            messagebox.showerror("Error", f"Could not save feedback: {e}")

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Prediction: ")
        self.stroke_history = []
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.current_prediction = None
        
        for canvas in self.aug_canvases:
            canvas.delete("all")
            
        self.playback_canvas.delete("all")

    def run(self):
        self.window.mainloop()

    def train_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model.train import train_and_save_model
        
        train_and_save_model(model_path)

if __name__ == "__main__":
    model_path = "models/mnist_model.keras"
    app = DigitRecognizer(model_path)
    app.run()