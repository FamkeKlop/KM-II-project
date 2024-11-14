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
import copy
from copy import deepcopy
import nibabel as nib


# Import custom modules
sys.path.append('functions/')
# from utils.demo import BboxPromptDemo, BboxPromptDemoTkinter
from segment_anything import sam_model_registry


class App:
    def __init__(self, root, window_name="GUI for Segmentation"):
        self.root = root
        self.root.title(window_name)
        self.root.geometry('800x500')
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
        """Load an image from an NRRD file and display the first slice."""
        file_path = filedialog.askopenfilename(title="Select NRRD file", filetypes=[("NRRD files", "*.nrrd"), ("Nifti files", "*.nii.gz")])
        
        if file_path:
            if file_path.endswith('.nrrd'):
                self.readdata, header = nrrd.read(file_path)
            elif file_path.endswith('.nii.gz'):
                nifit_image = nib.load(file_path)
                self.readdata = nifit_image.get_fdata()
                
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

            bbox_prompt_demo = BboxPromptDemo(self.medsam_model)
            bbox_prompt_demo.show(self.slice_rgb)


def show_mask(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

#ORIGINAL:
class BboxPromptDemo:
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
        self.fig, self.axes = None, None
        self.segs = []

    def _show(self, fig_size=5, random_color=True, alpha=0.65):
        assert self.image is not None, "Please set image first."

        self.fig, self.axes = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.resizable = False

        plt.tight_layout()
        self.axes.imshow(self.image)
        self.axes.axis('off')

        clear_button = widgets.Button(description="Clear", disabled=True)
        save_button = widgets.Button(description="Save", disabled=True)
        
        display(clear_button)
        display(save_button)

        def __on_press(event):
            if event.inaxes == self.axes:
                self.x0 = float(event.xdata) 
                self.y0 = float(event.ydata)
                self.currently_selecting = True
                self.rect = plt.Rectangle(
                    (self.x0, self.y0),
                    1,1, linestyle="--",
                    edgecolor="crimson", fill=False
                )
                self.axes.add_patch(self.rect)
                self.rect.set_visible(True)

        def __on_release(event):
            if event.inaxes == self.axes:
                if self.currently_selecting:
                    self.x1 = float(event.xdata)
                    self.y1 = float(event.ydata)
                    self.fig.canvas.draw_idle()
                    self.currently_selecting = False
                    self.rect.set_visible(False)
                    self.axes.patches[0].remove()
                    x_min = min(self.x0, self.x1)
                    x_max = max(self.x0, self.x1)
                    y_min = min(self.y0, self.y1)
                    y_max = max(self.y0, self.y1)
                    bbox = np.array([x_min, y_min, x_max, y_max])
                    with torch.no_grad():
                        seg = self._infer(bbox)
                        torch.cuda.empty_cache()
                    show_mask(seg, self.axes, random_color=random_color, alpha=alpha)
                    self.segs.append(deepcopy(seg))
                    del seg
                    self.rect = None
                    gc.collect()

                    save_button(disabled=False)

                    

        def __on_motion(event):
            if event.inaxes == self.axes:
                if self.currently_selecting:
                    self.x1 = float(event.xdata)
                    self.y1 = float(event.ydata)
                    #add rectangle for selection here
                    self.rect.set_visible(True)
                    xlim = np.sort([self.x0, self.x1])
                    ylim = np.sort([self.y0, self.y1])
                    self.rect.set_xy((xlim[0],ylim[0] ) )
                    rect_width = np.diff(xlim)[0]
                    self.rect.set_width(rect_width)
                    rect_height = np.diff(ylim)[0]
                    self.rect.set_height(rect_height)
                    self.fig.canvas.draw_idle()

        # clear_button = widgets.Button(description="Clear")
        def __on_clear_button_clicked(b):
            for i in range(len(self.axes.images)):
                self.axes.images[0].remove()
            self.axes.clear()
            self.axes.axis('off')
            self.axes.imshow(self.image)
            if len(self.axes.patches) > 0:
                self.axes.patches[0].remove()
            self.segs = []
            self.fig.canvas.draw_idle()

        def __on_save_button_clicked(b):
            plt.savefig("segmentation_results/seg_result.png", bbox_inches='tight', pad_inches=0)
            if len(self.segs) > 0:
                save_seg = np.zeros_like(self.segs[0])
                for i, seg in enumerate(self.segs, start=1):
                    save_seg[seg > 0] = i
                cv2.imwrite("segmentation_results/segs.png", save_seg)
                print(f"Segmentation result saved to {getcwd()}")
            
            # Save results as NRRD file
            nrrd.write("segmentation_results/segs.nrrd", save_seg)
        
        display(clear_button)
        clear_button.on_click(__on_clear_button_clicked)
        save_button.on_click(__on_save_button_clicked)


        self.fig.canvas.mpl_connect('button_press_event', __on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', __on_motion)
        self.fig.canvas.mpl_connect('button_release_event', __on_release)

        plt.show()

        # display(save_button)
        
    def show(self, image, fig_size=5, random_color=True, alpha=0.65):
        # Check if image is a string (file path) or a NumPy array
        if isinstance(image, str):
            # Use set_image_path only for file paths
            self.set_image_path(image)
        elif isinstance(image, np.ndarray):
            # Directly set the image if it's a NumPy array
            self._set_image(image)
        else:
            raise ValueError("Input must be a file path or a NumPy array.")
    
        self._show(fig_size=fig_size, random_color=random_color, alpha=alpha)

    def set_image_path(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._set_image(image)
    
    def _set_image(self, image):
        self.image = image
        self.img_size = image.shape[:2]
        image_preprocess = self._preprocess_image(image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)

    def _preprocess_image(self, image):
        img_resize = cv2.resize(
            image,
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC
        )
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        # convert the shape to (3, H, W)
        assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)

        return img_tensor
    
    @torch.no_grad()
    def _infer(self, bbox):
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
            image_embeddings = self.image_embeddings, # (B, 256, 64, 64)
            image_pe = self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
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
    

# Main section to start the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
