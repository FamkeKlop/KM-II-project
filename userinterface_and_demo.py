import gc
import torch
import numpy as np
import cv2
from torch.nn import functional as F
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from ipywidgets import interact, widgets, FileUpload
from IPython.display import display
from matplotlib import patches as patches
from matplotlib import pyplot as plt  # Import Matplotlib for plotting
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox, font as tkFont
from PIL import Image, ImageTk
import nrrd
import sys
import nibabel as nib
import copy

# Import custom modules
sys.path.append('functions/')
# from utils.demo import BboxPromptDemo, BboxPromptDemoTkinter
from segment_anything import sam_model_registry


class App:
    def __init__(self, root, window_name="GUI for Segmentation"):
        self.root = root
        self.root.title(window_name)
        self.root.geometry('900x500')
        self.image = None  # To store the loaded image
        self.readdata = None  # To store the loaded NRRD data
        self.slice_index = 0  # Initialize slice index
        self.current_plane = 0  # Initialize current plane index (0: axial, 1: coronal, 2: sagittal)
        self.current_image = None
        self.button_font = tkFont.Font(family="Helvetica", size=11)
        self.slice_rgb = None  # To store the specific slice for segmentation

        # Load the MedSAM model
        MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
        device = torch.device('cpu')  # Set to CPU for compatibility
        self.medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        self.medsam_model = self.medsam_model.to(device)
        self.medsam_model.eval()

        # Define plane labels
        self.planes = ['Axial', 'Coronal', 'Sagittal']

        # UI Elements
        self.setup_ui_elements()

    def setup_ui_elements(self):
        # Frame for the image and slider
        self.frame = tk.Frame(self.root)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas to display the image
        self.canvas = tk.Canvas(self.frame, width=500, height=400)
        self.canvas.pack(fill=tk.X, side=tk.TOP)

        # Slider for selecting slice index
        self.slice_slider = tk.Scale(self.frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_slice)
        self.slice_slider.pack()
        self.slice_slider.pack_forget()

        # Control frame for buttons and labels
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, padx=20, pady=20, expand=True)

        # Name of file_path title
        self.title_label = tk.Label(self.control_frame, font=('Helvetica', 11, 'bold'))
        self.title_label.pack(side='top', pady=(10, 5))

        # Load Image Button
        self.load_button = tk.Button(self.control_frame, text="Load NRRD or Nifti File", command=self.load_image_from_file, 
                                     bg='#abbdd9', font=self.button_font, width=18, height=2)
        self.load_button.pack(side=tk.TOP, pady=20)

        # Change Plane Button
        self.change_plane_button = tk.Button(self.control_frame, text="Change Plane", command=self.change_plane, 
                                             bg='#f5f3d5', font=self.button_font, width=18, height=2)
        self.change_plane_button.pack(pady=10)
        self.change_plane_button.pack_forget()  # Hide initially

        # Current Plane Label
        self.current_plane_label = tk.Label(self.control_frame, font=tkFont.Font(family="Helvetica", size=10, slant='italic'))
        self.current_plane_label.pack(pady=20)
        self.current_plane_label.pack_forget()  # Hide initially

        # Start Segmentation Button
        self.segmentation_button = tk.Button(self.control_frame, text="Start Segmentation", command=self.start_segmentation,
                                             bg='#b8cfb9', font=self.button_font, width=18, height=2)
        self.segmentation_button.pack()
        self.segmentation_button.pack_forget()

        # Quit Button
        self.quit_button = tk.Button(self.control_frame, text="Quit", command=self.root.quit, 
                                     bg='#3b3a3a', fg='#ffffff', font=self.button_font, width=18, height=2)
        self.quit_button.pack(side=tk.BOTTOM, pady=20)

    def load_image_from_file(self):
        """Load an image from an NRRD or NIFTI file and display the first slice."""
        file_path = filedialog.askopenfilename(title="Select NRRD file", filetypes=[("NRRD files", "*.nrrd"), ("NIFTI files", "*.nii.gz")])
        
        if file_path:
            if file_path.endswith('.nrrd'):
                self.readdata, header = nrrd.read(file_path)
            elif file_path.endswith('.nii.gz'):
                nifti_image = nib.load(file_path)
                self.readdata = nifti_image.get_fdata()
            
            # Extract file name for title and set title
            file_name = file_path.split('/')[-1]
            self.title_label.config(text=f'Image: {file_name}')
            
            self.update_slider_range()
            self.display_slice(self.slice_index)
            self.change_plane_button.pack()  # Show Change Plane button
            self.current_plane_label.pack()  # Show Current Plane label
            self.update_current_plane_label()  # Update plane label
            self.segmentation_button.pack(pady=20)
            self.slice_slider.pack(fill=tk.X, padx=20)

    def update_slider_range(self):
        """Update the slider range based on the current plane."""
        if self.current_plane == 0:
            self.slice_slider.config(from_=0, to=self.readdata.shape[2] - 1)
        elif self.current_plane == 1:
            self.slice_slider.config(from_=0, to=self.readdata.shape[1] - 1)
        elif self.current_plane == 2:
            self.slice_slider.config(from_=0, to=self.readdata.shape[0] - 1)
        self.slice_index = int(self.slice_slider.cget('to') / 2)  # Set to middle slice
        self.slice_slider.set(self.slice_index)

    def change_plane(self):
        """Change the current plane and update the slider range and displayed slice."""
        self.current_plane = (self.current_plane + 1) % len(self.planes)
        self.update_slider_range()
        self.display_slice(self.slice_index)
        self.update_current_plane_label()

    def update_current_plane_label(self):
        """Update the label to display the current plane."""
        self.current_plane_label.config(text=f"Current plane: {self.planes[self.current_plane]}")

    def update_slice(self, value):
        """Update the displayed slice when the slider is moved."""
        self.slice_index = int(value)
        self.display_slice(self.slice_index)

    def display_slice(self, slice_index):
        """Display the selected slice image on the Tkinter canvas."""
        if self.current_plane == 0:
            slice_image = self.readdata[:, :, slice_index]
        elif self.current_plane == 1:
            slice_image = self.readdata[:, slice_index, :]
        elif self.current_plane == 2:
            slice_image = self.readdata[slice_index, :, :]

        if len(slice_image.shape) == 2:
            slice_image = np.stack([slice_image] * 3, axis=-1)

        self.slice_rgb = self.normalize_to_uint8(slice_image)
        self.display_image(self.slice_rgb)

    def normalize_to_uint8(self, image):
        """Normalize a NumPy array (float image) to uint8."""
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_image = np.uint8(normalized_image)
        return normalized_image

    def display_image(self, rgb_image):
        """Display the loaded RGB image on the Tkinter canvas."""
        # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(rgb_image)
        self.current_image = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
        self.canvas.image = self.current_image

    def start_segmentation(self):
        if self.readdata is not None:
            # print(self.slice_rgb.shape)

            bbox_prompt_demo = BboxPromptDemoTkinter(self.medsam_model)
            bbox_prompt_demo.show(self.slice_rgb)

            # segmentation_window = tk.Toplevel(self.root)
            # segmentation_window.title("Segmentation")
            # segmentation_window.geometry("600x600")

            # canvas = tk.Canvas(segmentation_window, width=500, height=500)
            # canvas.pack()

            # bbox_demo = BboxPromptDemo(self.medsam_model)
            # bbox_demo.show(self.slice_rgb, canvas)

            # def start_draw(event):
            #     bbox_demo.start_x, bbox_demo.start_y = event.x, event.y
            #     if bbox_demo.rect:
            #         canvas.delete(bbox_demo.rect)
            #     bbox_demo.rect = canvas.create_rectangle(bbox_demo.start_x, bbox_demo.start_y, 
            #                                              bbox_demo.start_x, bbox_demo.start_y, outline="red")

            # def draw_rectangle(event):
            #     canvas.coords(bbox_demo.rect, bbox_demo.start_x, bbox_demo.start_y, event.x, event.y)

            # def end_draw(event):
            #     bbox_demo.end_x, bbox_demo.end_y = event.x, event.y
            #     bbox = [bbox_demo.start_x, bbox_demo.start_y, bbox_demo.end_x, bbox_demo.end_y]
            #     segmentation_mask = bbox_demo.get_mask(bbox)
            #     bbox_demo.show_mask(segmentation_mask)
            #     try_again_button.pack(side=tk.LEFT, padx=10, pady=10)
            #     save_mask_button.pack(side=tk.RIGHT, padx=10, pady=10)

            # def reset_mask():
            #     if bbox_demo.rect:
            #         canvas.delete(bbox_demo.rect)
            #     canvas.delete("all")
            #     bbox_demo.show(self.slice_rgb, canvas)
            #     try_again_button.pack_forget()
            #     save_mask_button.pack_forget()

            # def save_mask():
            #     self.saved_segmentation_mask = bbox_demo.segmentation_mask
            #     segmentation_window.destroy()

            # try_again_button = tk.Button(segmentation_window, text="Try Again", command=reset_mask)
            # save_mask_button = tk.Button(segmentation_window, text="Save Segmentation Mask", command=save_mask)

            # canvas.bind("<Button-1>", start_draw)
            # canvas.bind("<B1-Motion>", draw_rectangle)
            # canvas.bind("<ButtonRelease-1>", end_draw)

class BboxPromptDemoTkinter:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.gt = None
        self.currently_selecting = False
        self.x0, self.y0, self.x1, self.y1 = 0., 0., 0., 0.
        self.rect = None
        self.segs = []
        
        self.window = tk.Toplevel()
        self.window.title("Bounding Box Segmentation")
        
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.clear_button = tk.Button(self.window, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.LEFT)
        
        self.save_button = tk.Button(self.window, text="Save", command=self.save)
        self.save_button.pack(side=tk.LEFT)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _set_image(self, image):
        self.image = image
        self.img_size = image.shape[:2]
        image_preprocess = self._preprocess_image(image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)

        # Display the image in the Tkinter canvas
        self.display_image(image)

    def display_image(self, image):
        """Display the image on the Tkinter canvas."""
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = np.uint8(image)

        pil_image = Image.fromarray(image)
        tk_image = pil_image.resize((self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.tk_image = ImageTk.PhotoImage(tk_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image
        self.window.image = self.tk_image

    def on_press(self, event):
        """Called when the mouse button is pressed to start bounding box selection."""
        self.x0, self.y0 = event.x, event.y
        self.currently_selecting = True
        self.rect = self.canvas.create_rectangle(self.x0, self.y0, self.x0, self.y0, outline="crimson", width=2)

    def on_motion(self, event):
        """Called when the mouse is moved during bounding box selection."""
        if self.currently_selecting:
            self.x1, self.y1 = event.x, event.y
            self.canvas.coords(self.rect, self.x0, self.y0, self.x1, self.y1)

    def on_release(self, event):
        """Called when the mouse button is released to finalize bounding box."""
        if self.currently_selecting:
            self.x1, self.y1 = event.x, event.y
            self.currently_selecting = False
            
            scale_x = self.img_size[1] / self.canvas.winfo_width()
            scale_y = self.img_size[0] / self.canvas.winfo_height()
            
            # Draw and update the bounding box
            x_min = min(self.x0, self.x1) * scale_x
            x_max = max(self.x0, self.x1) * scale_x
            y_min = min(self.y0, self.y1) * scale_y
            y_max = max(self.y0, self.y1) * scale_y
            bbox = np.array([x_min, y_min, x_max, y_max])
            print(bbox)
            # Perform segmentation on the bounding box
            with torch.no_grad():
                seg = self._infer(bbox)
                torch.cuda.empty_cache()

            self.show_mask(seg)
            self.segs.append(copy.deepcopy(seg))
            self.rect = None
            gc.collect()

    def _infer(self, bbox):
        """Perform inference to generate the segmentation mask."""
        ori_H, ori_W = self.img_size
        scale_to_1024 = 1024 / np.array([ori_W, ori_H, ori_W, ori_H])
        bbox_1024 = bbox * scale_to_1024
        bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float).unsqueeze(0).to(self.model.device)
        if len(bbox_torch.shape) == 2:
            bbox_torch = bbox_torch.unsqueeze(1)
    
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=bbox_torch,
            masks=None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_pred,
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    def show_mask(self, mask, random_color=True, alpha=0.95):
        """Display the segmentation mask on the canvas."""
        
        if random_color:
            color = (np.random.randint(0, 256), 
                 np.random.randint(0, 256), 
                 np.random.randint(0, 256), 
                 int(alpha * 255))        
        else:
            color = (251, 252, 30, int(alpha * 255))  # Default yellowish color with alpha        
        
        h, w = mask.shape
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[..., :3] = np.array(color[:3])
        mask_rgba[..., 3] = (mask * color[3]).astype(np.uint8)
        
        mask_image = Image.fromarray(mask_rgba, mode="RGBA")
        
        img = Image.fromarray(self.image).convert("RGBA")
        combined = Image.alpha_composite(img, mask_image)
        
        tk_image = ImageTk.PhotoImage(combined.resize(
        (self.canvas.winfo_width(), self.canvas.winfo_height()),
        Image.Resampling.LANCZOS))
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.tk_image = tk_image
        
        # mask = Image.fromarray(mask * 255)
        # mask = mask.convert("RGBA")
        # img = Image.fromarray(self.image)
        # img = img.convert("RGBA")
        # img.paste(mask, (0, 0), mask)
        # self.tk_image = ImageTk.PhotoImage(img)
        # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def clear(self):
        """Clear the canvas and reset the selections."""
        self.canvas.delete("all")
        if self.image is not None:
            self.display_image(self.image)
        self.segs = []

    def save(self):
        """Save the segmentation results."""
        if self.segs:
            save_seg = np.zeros_like(self.segs[0])
            for i, seg in enumerate(self.segs, start=1):
                save_seg[seg > 0] = i
            cv2.imwrite("gen_mask/segs.png", save_seg)
            messagebox.showinfo("Saved", "Segmentation result saved to segs.png")
            
        # Save as NRRD file
        nrrd.write("segs.nrrd", save_seg)

    def show(self, image, random_color=True, alpha=0.65):
        """Display the image and run the segmentation demo."""
        if isinstance(image, str):
            self.set_image_path(image)
        elif isinstance(image, np.ndarray):
            self._set_image(image)
        else:
            raise ValueError("Input must be a file path or a NumPy array.")
        
        self.window.mainloop()

    def set_image_path(self, image_path):
        """Load image from file path and set."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._set_image(image)

    def _preprocess_image(self, image):
        """Preprocess the image for the model."""
        img_resize = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        return img_tensor

    def on_close(self):
        """Handle the window close event."""
        self.window.destroy()

# Main section to start the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()


