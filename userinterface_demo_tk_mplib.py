import gc
import torch
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import functional as F
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from ipywidgets import interact, widgets, FileUpload
from IPython.display import display
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox, font as tkFont
from PIL import Image, ImageTk
import nrrd
import sys
import copy
from copy import deepcopy

# Import custom modules
sys.path.append('functions/')
# from utils.demo import BboxPromptDemo, BboxPromptDemoTkinter
from segment_anything import sam_model_registry


class App:
    def __init__(self, root, window_name="GUI for Segmentation"):
        self.root = root
        self.root.title(window_name)
        self.root.geometry('850x600')
        self.image = None  # To store the loaded image
        self.readdata = None  # To store the loaded NRRD data
        self.slice_index = 0  # Initialize slice index
        self.current_plane = 0  # Initialize current plane index (0: axial, 1: coronal, 2: sagittal)
        self.current_image = None
        self.button_font = tkFont.Font(family="Helvetica", size=11)
        self.slice_rgb = None  # To store the specific slice for segmentation
        self.selected_organ = None
        self.begin_slice = None
        self.end_slice = None

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
        # Main Frame for padding and root structure
        self.main_frame = tk.Frame(self.root, padx=20, pady=10)  # Equal padding on both sides
        self.main_frame.grid(sticky='nsew')
        
        # Configure grid weights to allow expansion but keep the Quit button pinned in the bottom right
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)  # No extra weight for the last row with Quit button

        # Left frame for controls and labels, with more padding to center it between the window and the image
        self.left_frame = tk.Frame(self.main_frame, padx=10)  # Padding to center elements within this frame
        self.left_frame.grid(row=0, column=0, sticky='n', padx=(0, 20), pady=(0, 10))  # Left frame centered more

        # Load Image Button at the top left
        self.load_button = tk.Button(self.left_frame, text="Load NRRD or Nifti File",
                                    command=self.load_image_from_file, bg='#abbdd9',
                                    font=self.button_font, width=18, height=2)
        self.load_button.grid(row=0, column=0, pady=(10, 5), sticky="w")

        # File Path Title Label below the Load Button, centered
        self.title_label = tk.Label(self.left_frame, font=('Helvetica', 11, 'bold'))
        self.title_label.grid(row=1, column=0, pady=(0, 20), sticky="w")
        self.title_label.grid_remove()

        # Label for "Choose organ to segment"
        self.organ_label = tk.Label(self.left_frame, text="Choose organ to segment", font=('Helvetica', 10))
        self.organ_label.grid(row=2, column=0, pady=(5, 5), sticky="w")
        self.organ_label.grid_remove()

        # Text entry for organ input
        self.organ_entry = tk.Entry(self.left_frame, width=20)
        self.organ_entry.grid(row=3, column=0, pady=(0, 10), sticky="w")
        self.organ_entry.grid_remove()

        # 'Submit' button next to organ_entry
        self.submit_organ_button = tk.Button(self.left_frame, text="Submit", command=self.set_organ, height=1)
        self.submit_organ_button.grid(row=3, column=1, padx=(5, 0), sticky="w")
        self.submit_organ_button.grid_remove()

        # Label for "Selected organ is ..."
        self.selected_organ_label = tk.Label(self.left_frame, text="Selected organ is ", font=('Helvetica', 10))
        self.selected_organ_label.grid(row=4, column=0, pady=(5, 5), sticky="w")
        self.selected_organ_label.grid_remove()

        # Frame for "Begin" and "End" integer entries
        self.range_frame = tk.Frame(self.left_frame)
        self.range_frame.grid(row=5, column=0, sticky="w", pady=(5, 10))

        # Begin Label and Entry
        self.begin_label = tk.Label(self.range_frame, text="Begin slice", font=('Helvetica', 9))
        self.begin_label.grid(row=0, column=0, padx=(0, 10), sticky="w")
        self.begin_label.grid_remove()
        self.begin_entry = tk.Entry(self.range_frame, width=5)
        self.begin_entry.grid(row=1, column=0, padx=(0, 10), sticky="w")
        self.begin_entry.grid_remove()

        # End Label and Entry
        self.end_label = tk.Label(self.range_frame, text="End slice", font=('Helvetica', 9))
        self.end_label.grid(row=0, column=1, padx=(10, 0), sticky="w")
        self.end_label.grid_remove()
        self.end_entry = tk.Entry(self.range_frame, width=5)
        self.end_entry.grid(row=1, column=1, padx=(10, 0), sticky="w")
        self.end_entry.grid_remove()

        # 'Submit' button for slice range
        self.submit_range_button = tk.Button(self.range_frame, text="Submit", command=self.set_organ_range, height=1)
        self.submit_range_button.grid(row=1, column=2, padx=(5, 0), sticky="w")
        self.submit_range_button.grid_remove()

        # Label for "Slice range of organ is ..."
        self.organ_range_label = tk.Label(self.left_frame, text="Slice range of organ is ", font=('Helvetica', 10))
        self.organ_range_label.grid(row=6, column=0, pady=(5, 5), sticky="w")
        self.organ_range_label.grid_remove()

        # Start Segmentation Button aligned to the bottom left
        self.segmentation_button = tk.Button(self.left_frame, text="Start Segmentation",
                                            command=self.start_segmentation,
                                            bg='#b8cfb9', font=self.button_font, width=18, height=2)
        self.segmentation_button.grid(row=7, column=0, pady=(5, 20), sticky="sw")
        self.segmentation_button.grid_remove()

        # Right frame for image and related controls
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=1, sticky='ne', padx=(10, 20), pady=(10, 10))  # Adjusted right padding

        # Canvas to display the image
        self.canvas = tk.Canvas(self.image_frame, width=500, height=400)
        self.canvas.grid(row=0, column=0, sticky='n')

        # Frame for slider and plane control buttons, below the image
        self.slider_control_frame = tk.Frame(self.image_frame)
        self.slider_control_frame.grid(row=1, column=0, sticky='ew', pady=(10, 5))

        # Change Plane Button on the left in slider control frame
        self.change_plane_button = tk.Button(self.slider_control_frame, text="Change Plane",
                                            command=self.change_plane, bg='#D3D3D3',
                                            font=self.button_font, width=10, height=1)
        self.change_plane_button.grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.change_plane_button.grid_remove()

        # Slider, next to Change Plane button, with a larger width to reach the right side
        self.slice_slider = tk.Scale(self.slider_control_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                    command=self.update_slice, length=400)
        self.slice_slider.grid(row=0, column=1, sticky="e")
        self.slice_slider.grid_remove()

        # Current Plane Label below the slider and change plane button
        self.current_plane_label = tk.Label(self.image_frame,
                                            font=tkFont.Font(family="Helvetica", size=10, slant='italic'))
        self.current_plane_label.grid(row=2, column=0, pady=(5, 20), sticky="n")
        self.current_plane_label.grid_remove()

        # Quit Button at the right bottom corner
        self.quit_button = tk.Button(self.main_frame, text="Quit", command=self.root.quit,
                                    bg='#3b3a3a', fg='#ffffff', font=self.button_font, width=18, height=2)
        self.quit_button.grid(row=1, column=1, sticky='se', padx=(0, 20), pady=(0, 10))


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
            self.update_current_plane_label()  # Update plane label
            self.title_label.grid()
            self.change_plane_button.grid()
            self.slice_slider.grid()
            self.current_plane_label.grid()
            self.organ_label.grid()
            self.organ_entry.grid()
            self.submit_organ_button.grid()


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

    def set_organ(self):
        """Update the selected organ label with the value from the organ_entry."""
        self.selected_organ = self.organ_entry.get()
        new_text = 'Selected organ is ' + self.selected_organ
        self.selected_organ_label.config(text=new_text)
        self.selected_organ_label.grid()
        self.begin_label.grid()
        self.begin_entry.grid()
        self.end_label.grid()
        self.end_entry.grid()
        self.submit_range_button.grid()
        self.organ_range_label.grid_remove()

        # Remove previous information
        self.begin_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)
        self.begin_slice = None
        self.end_slice = None
        self.segmentation_button.grid_remove()


    def set_organ_range(self):
        self.begin_slice = self.begin_entry.get()
        self.end_slice = self.end_entry.get()

        new_text = "Slice range of " + self.selected_organ + ' is [' + self.begin_slice + ', ' + self.end_slice + ']'
        self.organ_range_label.config(text=new_text)
        self.organ_range_label.grid()

        self.segmentation_button.grid()
 
    def start_segmentation(self):
        if self.readdata is not None:
            middle_slice = int((float(self.end_slice) - float(self.begin_slice))/2 + float(self.begin_slice))

            self.slice_index = middle_slice
            self.slice_slider.set(self.slice_index)
            self.display_slice(self.slice_index)

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

        clear_ax = self.fig.add_axes([0.1, 0.05, 0.3, 0.075])
        save_ax = self.fig.add_axes([0.7, 0.05, 0.2, 0.075])

        clear_button = matplotlib.widgets.Button(clear_ax, 'Retry segmentation')
        save_button = matplotlib.widgets.Button(save_ax, "Save mask")

        clear_button.ax.set_visible(False)
        save_button.ax.set_visible(False)

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

                    clear_button.ax.set_visible(True)
                    save_button.ax.set_visible(True)

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

        def __on_clear_button_clicked(event):
            for i in range(len(self.axes.images)):
                self.axes.images[0].remove()
            self.axes.clear()
            self.axes.axis('off')
            self.axes.imshow(self.image)
            if len(self.axes.patches) > 0:
                self.axes.patches[0].remove()
            self.segs = []
            self.fig.canvas.draw_idle()
            clear_button.ax.set_visible(False)
            save_button.ax.set_visible(False)

        def __on_save_button_clicked(event):
            plt.savefig("seg_result.png", bbox_inches='tight', pad_inches=0)
            if len(self.segs) > 0:
                save_seg = np.zeros_like(self.segs[0])
                for i, seg in enumerate(self.segs, start=1):
                    save_seg[seg > 0] = i
                cv2.imwrite("segs.png", save_seg)
                print(f"Segmentation result saved to {getcwd()}")

        clear_button.on_clicked(__on_clear_button_clicked)
        save_button.on_clicked(__on_save_button_clicked)


        self.fig.canvas.mpl_connect('button_press_event', __on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', __on_motion)
        self.fig.canvas.mpl_connect('button_release_event', __on_release)

        plt.show()
        
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
