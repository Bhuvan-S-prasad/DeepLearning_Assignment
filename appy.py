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
import os

COLORS = {
    'bg_main': '#1e1e2e',       
    'bg_sidebar': '#262939',    
    'card_bg': '#313244',       
    'accent': '#89b4fa',        
    'success': '#a6e3a1',       
    'warning': '#f9e2af',      
    'danger': '#f38ba8',        
    'text': '#cdd6f4',          
    'subtext': '#a6adc8',       
    'button_def': '#45475a',    
    'button_hover': '#585b70'   
}

FONTS = {
    'header': ("Segoe UI", 20, "bold"),
    'subheader': ("Segoe UI", 14, "bold"),
    'body': ("Segoe UI", 11),
    'body_bold': ("Segoe UI", 11, "bold"),
    'small': ("Segoe UI", 9)
}

class ModernButton(tk.Button):
    """Custom Button with hover effects"""
    def __init__(self, master, **kw):
        self.bg_color = kw.get('bg', COLORS['button_def'])
        self.hover_color = kw.get('activebackground', COLORS['button_hover'])
        kw['relief'] = tk.FLAT
        kw['borderwidth'] = 0
        kw['cursor'] = 'hand2'
        kw['activeforeground'] = kw.get('fg', 'white')
        super().__init__(master, **kw)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = self.hover_color

    def on_leave(self, e):
        self['background'] = self.bg_color

class DenseNetBrainTumorClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroScan AI | DenseNet Brain Tumor Classifier")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg_main'])
        
        # Logic Variables
        self.class_names = ['glioma', 'healthy', 'meningioma', 'pituitary']
        self.confidence_threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.cam = None
        self.current_image_path = None
        self.display_size = (350, 350)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.setup_styles()
        self.setup_gui()
        self.load_model()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Progress Bar
        style.configure("Horizontal.TProgressbar", 
                        troughcolor=COLORS['button_def'], 
                        background=COLORS['accent'],
                        lightcolor=COLORS['accent'],
                        darkcolor=COLORS['accent'],
                        bordercolor=COLORS['card_bg'],
                        thickness=10)

    def setup_gui(self):
        # --- Sidebar ---
        sidebar = tk.Frame(self.root, bg=COLORS['bg_sidebar'], width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

      
        tk.Label(sidebar, text="🧠 NeuroScan AI", font=FONTS['header'], 
                 bg=COLORS['bg_sidebar'], fg=COLORS['accent']).pack(pady=(40, 10), padx=20, anchor="w")
        tk.Label(sidebar, text="Deep Learning Diagnostics", font=FONTS['small'], 
                 bg=COLORS['bg_sidebar'], fg=COLORS['subtext']).pack(pady=(0, 30), padx=20, anchor="w")

        # Control Buttons
        btn_frame = tk.Frame(sidebar, bg=COLORS['bg_sidebar'])
        btn_frame.pack(fill=tk.X, padx=20)

        self.upload_btn = ModernButton(
            btn_frame, text="📁  Upload MRI Scan", command=self.upload_image,
            bg=COLORS['accent'], fg=COLORS['bg_main'], activebackground="#b4befe",
            font=FONTS['body_bold'], height=2
        )
        self.upload_btn.pack(fill=tk.X, pady=10)

        self.predict_btn = ModernButton(
            btn_frame, text="⚡  Analyze Scan", command=self.predict,
            bg=COLORS['button_def'], fg="white", activebackground=COLORS['success'],
            font=FONTS['body_bold'], height=2, state=tk.DISABLED
        )
        self.predict_btn.pack(fill=tk.X, pady=10)

        # Status Footer
        self.status_label = tk.Label(sidebar, text="System Ready", font=FONTS['small'],
                                   bg=COLORS['bg_sidebar'], fg=COLORS['subtext'], anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        # --- Main Content ---
        main_content = tk.Frame(self.root, bg=COLORS['bg_main'])
        main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=40, pady=40)

        # 1. Visualization Area 
        vis_frame = tk.Frame(main_content, bg=COLORS['bg_main'])
        vis_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original Image Card
        self.card_original = self.create_image_card(vis_frame, "Original MRI Scan")
        self.card_original.pack(side=tk.LEFT, padx=(0, 20), expand=True)
        self.lbl_original = self.card_original.image_label

        # Grad-CAM Card
        self.card_gradcam = self.create_image_card(vis_frame, "AI Attention Map (Grad-CAM)")
        self.card_gradcam.pack(side=tk.LEFT, padx=(20, 0), expand=True)
        self.lbl_gradcam = self.card_gradcam.image_label

        # 2. Results Area 
        results_container = tk.Frame(main_content, bg=COLORS['card_bg'], pady=20, padx=20)
        results_container.pack(fill=tk.X, pady=(30, 0))

        # Prediction Header
        res_header = tk.Frame(results_container, bg=COLORS['card_bg'])
        res_header.pack(fill=tk.X, pady=(0, 15))
        
        self.lbl_pred_class = tk.Label(res_header, text="Waiting for input...", font=("Segoe UI", 24, "bold"),
                                     bg=COLORS['card_bg'], fg=COLORS['subtext'])
        self.lbl_pred_class.pack(side=tk.LEFT)
        
        self.lbl_confidence = tk.Label(res_header, text="", font=("Segoe UI", 18),
                                     bg=COLORS['card_bg'], fg=COLORS['accent'])
        self.lbl_confidence.pack(side=tk.RIGHT)

        ttk.Separator(results_container, orient='horizontal').pack(fill=tk.X, pady=15)

        probs_frame = tk.Frame(results_container, bg=COLORS['card_bg'])
        probs_frame.pack(fill=tk.X)

        self.prob_bars = {}
        self.prob_labels = {}
        
        for idx, class_name in enumerate(self.class_names):
            col = idx % 2
            row = idx // 2
            
            p_frame = tk.Frame(probs_frame, bg=COLORS['card_bg'])
            p_frame.grid(row=row, column=col, sticky="ew", padx=20, pady=10)
            probs_frame.columnconfigure(col, weight=1)
            
            tk.Label(p_frame, text=class_name.capitalize(), font=FONTS['body'],
                     bg=COLORS['card_bg'], fg=COLORS['text'], width=12, anchor='w').pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(p_frame, length=100, mode='determinate', style="Horizontal.TProgressbar")
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            
            perc = tk.Label(p_frame, text="0%", font=FONTS['body_bold'],
                          bg=COLORS['card_bg'], fg=COLORS['subtext'], width=5)
            perc.pack(side=tk.RIGHT)
            
            self.prob_bars[class_name] = bar
            self.prob_labels[class_name] = perc

        self.set_placeholder_images()

    def create_image_card(self, parent, title):
        """Helper to create a styled image card"""
        card = tk.Frame(parent, bg=COLORS['card_bg'], padx=10, pady=10)
        
        # Header
        tk.Label(card, text=title, font=FONTS['subheader'], 
                 bg=COLORS['card_bg'], fg=COLORS['text']).pack(pady=(5, 15))
        
        # Image Container 
        img_container = tk.Frame(card, bg="black", bd=2, relief=tk.FLAT)
        img_container.pack(expand=True)
        
        lbl = tk.Label(img_container, bg="black")
        lbl.pack()
        
        card.image_label = lbl
        return card

    def set_placeholder_images(self):
        img = Image.new('RGB', self.display_size, color='#252630')
        draw = ImageDraw.Draw(img)
        
        cx, cy = self.display_size[0] // 2, self.display_size[1] // 2
        draw.line((cx-20, cy, cx+20, cy), fill='#45475a', width=3)
        draw.line((cx, cy-20, cx, cy+20), fill='#45475a', width=3)
        
        photo = ImageTk.PhotoImage(img)
        self.lbl_original.config(image=photo)
        self.lbl_original.image = photo
        self.lbl_gradcam.config(image=photo)
        self.lbl_gradcam.image = photo

    def load_model(self):
        self.status_label.config(text="⏳ Loading Neural Network...")
        self.root.update()
        try:
            # Model Definition
            model = models.densenet121(weights="IMAGENET1K_V1")
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, len(self.class_names))
            )
            
            # Weights
            if os.path.exists('dense_model.pth'):
                model.load_state_dict(torch.load('dense_model.pth', map_location=self.device, weights_only=True))
                model = model.to(self.device)
                model.eval()
                
                target_layers = [model.features.norm5]
                self.cam = GradCAM(model=model, target_layers=target_layers)
                self.model = model
                self.status_label.config(text="System Ready - Model Loaded")
            else:
                self.status_label.config(text="Model file not found (Demo Mode)")
                messagebox.showwarning("Model Missing", "dense_model.pth not found.\nPlease ensure the model file is in the directory.")
                
        except Exception as e:
            self.status_label.config(text="Model Error")
            print(e)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("MRI Images", "*.png *.jpg *.jpeg *.tif *.bmp")])
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path, self.lbl_original)
            
            # Reset UI
            self.set_placeholder_gradcam()
            self.predict_btn.config(state=tk.NORMAL, bg=COLORS['success'])
            self.lbl_pred_class.config(text="Ready to Analyze", fg=COLORS['text'])
            self.lbl_confidence.config(text="")
            self.status_label.config(text=f"Image Loaded: {os.path.basename(file_path)}")
            
            # Reset Bars
            for bar in self.prob_bars.values(): bar['value'] = 0
            for lbl in self.prob_labels.values(): lbl.config(text="0%")

    def set_placeholder_gradcam(self):
        img = Image.new('RGB', self.display_size, color='#252630')
        draw = ImageDraw.Draw(img)
        cx, cy = self.display_size[0] // 2, self.display_size[1] // 2
        draw.text((cx-50, cy), "Analysis Pending", fill='#45475a')
        photo = ImageTk.PhotoImage(img)
        self.lbl_gradcam.config(image=photo)
        self.lbl_gradcam.image = photo

    def display_image(self, path_or_img, label_widget):
        if isinstance(path_or_img, str):
            img = Image.open(path_or_img).convert('RGB')
        else:
            img = path_or_img
            
        img.thumbnail(self.display_size, Image.Resampling.LANCZOS)
        bg_img = Image.new('RGB', self.display_size, (0, 0, 0))
        offset = ((self.display_size[0] - img.size[0]) // 2, (self.display_size[1] - img.size[1]) // 2)
        bg_img.paste(img, offset)
        
        photo = ImageTk.PhotoImage(bg_img)
        label_widget.config(image=photo)
        label_widget.image = photo

    def predict(self):
        if not self.model or not self.current_image_path: return
        
        try:
            self.status_label.config(text="Processing...")
            self.root.update()
            
            image = Image.open(self.current_image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
            
            conf = float(probs[pred_idx])
            pred_name = self.class_names[pred_idx]
            
            # Update Text
            color = COLORS['success'] if conf > 0.7 else COLORS['warning']
            if conf < 0.5: color = COLORS['danger']
            
            self.lbl_pred_class.config(text=pred_name.upper(), fg=color)
            self.lbl_confidence.config(text=f"{conf:.1%}", fg=color)
            
            # Update Bars
            for i, name in enumerate(self.class_names):
                val = float(probs[i])
                self.prob_bars[name]['value'] = val * 100
                self.prob_labels[name].config(text=f"{val:.1%}")
                
            # Grad-CAM
            self.generate_gradcam(image, tensor, pred_idx)
            self.status_label.config(text="Analysis Complete")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Error")

    def generate_gradcam(self, original_img, tensor, target_class):
        if self.cam is None: return
        
        grayscale_cam = self.cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_class)])[0, :]
        img_resized = original_img.resize((224, 224))
        img_np = np.array(img_resized)
        
        vis = show_cam_on_image(img_np / 255.0, grayscale_cam, use_rgb=True)
        pil_vis = Image.fromarray(vis)
        self.display_image(pil_vis, self.lbl_gradcam)

if __name__ == "__main__":
    root = tk.Tk()
    app = DenseNetBrainTumorClassifier(root)
    root.mainloop()