import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import io

class DenseNetBrainTumorClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("DenseNet Brain Tumor Classifier")
        # Adjusted geometry to fit standard screens better (reduced height)
        self.root.geometry("1300x850")
        self.root.resizable(True, True)
        
        # Configure root window to expand properly
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Configuration
        self.class_names = ['glioma', 'healthy', 'meningioma', 'pituitary']
        self.confidence_threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # UI Configuration
        self.display_size = (300, 300)  # Reduced from 400x400 to save vertical space
        
        self.model = None
        self.cam = None
        self.current_image_path = None
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Setup GUI
        self.setup_gui()
        
        # Load model
        self.load_model()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Title - Reduced padding
        title_label = tk.Label(
            self.root, 
            text="DenseNet Brain Tumor Classifier", 
            font=("Helvetica", 16, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=10
        )
        title_label.pack(fill=tk.X)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#ecf0f1")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#ecf0f1")
        button_frame.pack(pady=5)
        
        # Upload button
        self.upload_btn = tk.Button(
            button_frame,
            text="Upload MRI Image",
            command=self.upload_image,
            font=("Helvetica", 11),
            bg="#3498db",
            fg="white",
            padx=15,
            pady=5,
            cursor="hand2"
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Predict button
        self.predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            font=("Helvetica", 11),
            bg="#27ae60",
            fg="white",
            padx=15,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display frame - centered
        image_frame = tk.Frame(main_frame, bg="#ecf0f1")
        image_frame.pack(pady=5, fill=tk.BOTH, expand=False)
        
        # Container for centering the two images
        center_image_container = tk.Frame(image_frame, bg="#ecf0f1")
        center_image_container.pack(anchor="center")

        # Original image Frame
        original_frame = tk.LabelFrame(
            center_image_container, 
            text="Original Image", 
            font=("Helvetica", 10, "bold"),
            bg="#ecf0f1"
        )
        original_frame.pack(side=tk.LEFT, padx=20)
        
        self.original_image_label = tk.Label(
            original_frame, 
            bg="white"
        )
        self.original_image_label.pack(padx=5, pady=5)
        
        # Grad-CAM visualization Frame
        gradcam_frame = tk.LabelFrame(
            center_image_container, 
            text="Grad-CAM Visualization", 
            font=("Helvetica", 10, "bold"),
            bg="#ecf0f1"
        )
        gradcam_frame.pack(side=tk.LEFT, padx=20)
        
        self.gradcam_image_label = tk.Label(
            gradcam_frame, 
            bg="white"
        )
        self.gradcam_image_label.pack(padx=5, pady=5)
        
        # Set placeholder images to fix layout size immediately
        self.set_placeholder_images()

        # Results frame
        results_frame = tk.LabelFrame(
            main_frame, 
            text="Prediction Results", 
            font=("Helvetica", 12, "bold"),
            bg="#ecf0f1"
        )
        results_frame.pack(fill=tk.X, pady=10, padx=10, side=tk.BOTTOM, anchor="s")
        
        # Predicted class and Confidence on one line (optional) or tight stacking
        summary_frame = tk.Frame(results_frame, bg="#ecf0f1")
        summary_frame.pack(fill=tk.X, pady=5)

        self.class_label = tk.Label(
            summary_frame,
            text="Predicted Class: -",
            font=("Helvetica", 14, "bold"),
            bg="#ecf0f1"
        )
        self.class_label.pack(side=tk.LEFT, padx=20)
        
        self.confidence_label = tk.Label(
            summary_frame,
            text="Confidence: -",
            font=("Helvetica", 12),
            bg="#ecf0f1"
        )
        self.confidence_label.pack(side=tk.LEFT, padx=20)
        
        # Probabilities
        prob_frame = tk.Frame(results_frame, bg="#ecf0f1")
        prob_frame.pack(pady=5, fill=tk.X, padx=20)
        
        self.prob_labels = {}
        self.prob_bars = {}
        
        for class_name in self.class_names:
            row_frame = tk.Frame(prob_frame, bg="#ecf0f1")
            row_frame.pack(fill=tk.X, pady=2) # Reduced vertical padding
            
            label = tk.Label(
                row_frame,
                text=f"{class_name.capitalize()}:",
                font=("Helvetica", 10),
                bg="#ecf0f1",
                width=15,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(
                row_frame,
                length=300, # Slightly shorter bars
                mode='determinate'
            )
            progress.pack(side=tk.LEFT, padx=10)
            
            percent_label = tk.Label(
                row_frame,
                text="0%",
                font=("Helvetica", 10),
                bg="#ecf0f1",
                width=8
            )
            percent_label.pack(side=tk.LEFT)
            
            self.prob_bars[class_name] = progress
            self.prob_labels[class_name] = percent_label
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Helvetica", 9),
            bg="#34495e",
            fg="white",
            anchor='w',
            padx=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def set_placeholder_images(self):
        """Create a blank gray image to hold the layout size fixed"""
        img = Image.new('RGB', self.display_size, color='#bdc3c7')
        draw = ImageDraw.Draw(img)
        # Draw a subtle cross or text to indicate placeholder
        w, h = self.display_size
        draw.line((0, 0, w, h), fill='#95a5a6', width=2)
        draw.line((0, h, w, 0), fill='#95a5a6', width=2)
        
        photo = ImageTk.PhotoImage(img)
        
        self.original_image_label.config(image=photo)
        self.original_image_label.image = photo
        
        self.gradcam_image_label.config(image=photo)
        self.gradcam_image_label.image = photo

    def load_model(self):
        """Load the DenseNet model"""
        try:
            self.status_label.config(text="Loading DenseNet model...")
            self.root.update()
            
            # Load DenseNet-121
            model = models.densenet121(weights="IMAGENET1K_V1")
            
            # Modify classifier
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features, len(self.class_names))
            )
            
            # Load trained weights
            # Ensure 'dense_model.pth' is in the same directory or provide full path
            try:
                model.load_state_dict(
                    torch.load('dense_model.pth', map_location=self.device, weights_only=True)
                )
            except FileNotFoundError:
                messagebox.showerror("Error", "Model file 'dense_model.pth' not found in directory.")
                return

            model = model.to(self.device)
            model.eval()
            
            # Setup GradCAM
            # For DenseNet, features.norm5 is the last layer of features before classifier
            target_layers = [model.features.norm5]
            self.cam = GradCAM(model=model, target_layers=target_layers)
            
            self.model = model
            self.status_label.config(text="Model loaded successfully")
            
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_original_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Image loaded: {file_path}")
            
            # Reset results, but keep the original image displayed
            # Reset GradCAM to placeholder
            self.reset_gradcam_placeholder()
            
            self.class_label.config(text="Predicted Class: -", fg="black")
            self.confidence_label.config(text="Confidence: -", fg="black")
            for class_name in self.class_names:
                self.prob_bars[class_name]['value'] = 0
                self.prob_labels[class_name].config(text="0%")

    def reset_gradcam_placeholder(self):
        img = Image.new('RGB', self.display_size, color='#bdc3c7')
        photo = ImageTk.PhotoImage(img)
        self.gradcam_image_label.config(image=photo)
        self.gradcam_image_label.image = photo
    
    def display_original_image(self, image_path):
        """Display the original image"""
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize using thumbnail to maintain aspect ratio, but fit within box
            image.thumbnail(self.display_size, Image.Resampling.LANCZOS)
            
            # Create a background of fixed size to ensure UI stability
            bg_img = Image.new('RGB', self.display_size, (255, 255, 255))
            
            # Center the thumbnail on the background
            offset = ((self.display_size[0] - image.size[0]) // 2,
                      (self.display_size[1] - image.size[1]) // 2)
            bg_img.paste(image, offset)
            
            photo = ImageTk.PhotoImage(bg_img)
            self.original_image_label.config(image=photo)
            self.original_image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image:\n{str(e)}")
    
    def predict(self):
        """Make prediction on the uploaded image"""
        if not self.current_image_path or not self.model:
            return
        
        try:
            self.status_label.config(text="Making prediction...")
            self.root.update()
            
            # Load and preprocess image
            image = Image.open(self.current_image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
            
            max_confidence = float(probabilities[predicted_class])
            
            # Update results
            if max_confidence < self.confidence_threshold:
                self.class_label.config(
                    text="Predicted Class: Unknown",
                    fg="#e74c3c"
                )
                self.confidence_label.config(
                    text=f"Confidence: {max_confidence:.2%}",
                    fg="#e74c3c"
                )
            else:
                self.class_label.config(
                    text=f"Predicted Class: {self.class_names[predicted_class].upper()}",
                    fg="#27ae60"
                )
                self.confidence_label.config(
                    text=f"Confidence: {max_confidence:.2%}",
                    fg="#27ae60"
                )
            
            # Update probability bars
            for i, class_name in enumerate(self.class_names):
                prob = float(probabilities[i])
                self.prob_bars[class_name]['value'] = prob * 100
                self.prob_labels[class_name].config(text=f"{prob:.1%}")
            
            # Generate and display Grad-CAM
            self.generate_gradcam(image, image_tensor, predicted_class)
            
            self.status_label.config(text="Prediction complete")
            
        except Exception as e:
            self.status_label.config(text=f"Error during prediction: {str(e)}")
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
    
    def generate_gradcam(self, original_image, image_tensor, target_class):
        """Generate and display Grad-CAM visualization"""
        try:
            if self.cam is None:
                return
            
            # Generate Grad-CAM
            grayscale_cam = self.cam(
                input_tensor=image_tensor,
                targets=[ClassifierOutputTarget(target_class)]
            )
            grayscale_cam = grayscale_cam[0, :]
            
            # Resize original image to 224x224 for Grad-CAM processing (Model Input Size)
            img_resized = original_image.resize((224, 224))
            img_np = np.array(img_resized)
            
            # Create visualization
            visualization = show_cam_on_image(
                img_np / 255.0,
                grayscale_cam,
                use_rgb=True,
                colormap=2
            )
            
            # Convert back to PIL
            pil_img = Image.fromarray(visualization)
            
            # Resize to match the display size (300x300)
            pil_img.thumbnail(self.display_size, Image.Resampling.LANCZOS)
            
            # Center on background like the original image
            bg_img = Image.new('RGB', self.display_size, (255, 255, 255))
            offset = ((self.display_size[0] - pil_img.size[0]) // 2,
                      (self.display_size[1] - pil_img.size[1]) // 2)
            bg_img.paste(pil_img, offset)
            
            photo = ImageTk.PhotoImage(bg_img)
            self.gradcam_image_label.config(image=photo)
            self.gradcam_image_label.image = photo
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            self.status_label.config(text=f"Grad-CAM generation failed: {str(e)}")

def main():
    root = tk.Tk()
    app = DenseNetBrainTumorClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()