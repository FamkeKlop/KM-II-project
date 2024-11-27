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
import nibabel as nib
from tkinter import ttk
import time

# Import custom modules
sys.path.append('functions/')
# from utils.demo import BboxPromptDemo, BboxPromptDemoTkinter
from segment_anything import sam_model_registry

class BboxPromptDemo:
    def __init__(self, model, data, begin_slice, end_slice, plane, slice_index, master, callback, segmented_organs_list, selected_organ):
        
        self.model = model
        self.model.eval()
        self.readdata = data
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.begin_slice = begin_slice
        self.end_slice = end_slice
        self.plane = plane
        self.gt = None
        self.currently_selecting = False
        self.x0, self.y0, self.x1, self.y1 = 0., 0., 0., 0.
        self.rect = None
        self.segs = []
        self.bbox = None
        self.master = master
        self.callback = callback
        self.segmented_organs_list = segmented_organs_list
        self.selected_organ = selected_organ
        
        self.window = tk.Toplevel(master)
        self.window.title("Bounding Box Segmentation")
        
        # Set image
        initial_slice = self.get_image_from_plane(self.plane, slice_index)
        initial_slice = self.normalize_to_uint8(initial_slice)

        img_height, img_width = initial_slice.shape[:2]

        self.canvas = tk.Canvas(self.window, width=img_width, height=img_height)
        self.canvas.grid(row=0, column=0, columnspan=3,sticky='nsew')
        
        self.progressbar = ttk.Progressbar(self.window, orient='horizontal', length=100)
        self.progressbar.grid(row=3, column=0, columnspan=3, sticky='nsew')
        self.progressbar.grid_remove()
        
        self.clear_button = tk.Button(self.window, text="Try again", command=self.clear)
        self.clear_button.grid(row=1, column=0,pady=10,padx=10)
        self.clear_button.grid_remove()
        
        self.save_button = tk.Button(self.window, text="Segment for all slices in organ range", command=self.save)
        self.save_button.grid(row=1, column=1, pady=10,padx=10)
        self.save_button.grid_remove()


        self._set_image(initial_slice)
        
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Bind mouse events        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def _set_image(self, image):
        """ Prepares the image for the model by preprocessing it and extracting the embedding. The
            image is displayed in the Tkinter canvas.
            Args:
                image: 2D NumPy array or 3D NumPy array (RGB image).
            Calls:
                _preprocess_image(), display_image()
        """
        self.image = image
        self.img_size = image.shape[:2]
        image_preprocess = self._preprocess_image(image)
        
        # Get image embedding from encoder
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)

        # Display the image in the Tkinter canvas
        self.display_image(image)

    def normalize_to_uint8(self, image):
        """Normalize a NumPy array (float image) to uint8.
            Args:
                image: 2D NumPy array or 3D NumPy array (RGB image).
            Returns:
                normalized_image: 2D NumPy array or 3D NumPy array"""

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_image = np.uint8(normalized_image)
        return normalized_image

    def display_image(self, image):
        """Display the image on the Tkinter canvas.
            Args:
                image: 2D NumPy array or 3D NumPy array (RGB image)."""
        if image.dtype != np.uint8:
            image = self.normalize_to_uint8(image)

        self.canvas.delete("all")

        pil_image = Image.fromarray(image)
        tk_image = pil_image.resize((self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.tk_image = ImageTk.PhotoImage(tk_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image
        self.window.image = self.tk_image

    def on_press(self, event):
        """Called when the mouse button is pressed to start bounding box selection.
            Args:
                event: mouse event (clicking)."""
        self.x0, self.y0 = event.x, event.y
        self.currently_selecting = True
        self.rect = self.canvas.create_rectangle(self.x0, self.y0, self.x0, self.y0, outline="crimson", width=2)

    def on_motion(self, event):
        """Called when the mouse is moved during bounding box selection.
            Args:
                event: mouse event (moving)."""
        if self.currently_selecting:
            self.x1, self.y1 = event.x, event.y
            self.canvas.coords(self.rect, self.x0, self.y0, self.x1, self.y1)

    def on_release(self, event):
        """Called when the mouse button is released to finalize bounding box.
            Args:  
                event: mouse event (release)."""
        if self.currently_selecting:
            self.x1, self.y1 = event.x, event.y
            self.currently_selecting = False
            
            # Scale to from pixel space to image space
            scale_x = self.img_size[1] / self.canvas.winfo_width()
            scale_y = self.img_size[0] / self.canvas.winfo_height()
            
            # Draw and update the bounding box
            x_min = min(self.x0, self.x1) * scale_x
            x_max = max(self.x0, self.x1) * scale_x
            y_min = min(self.y0, self.y1) * scale_y
            y_max = max(self.y0, self.y1) * scale_y
            self.bbox = np.array([x_min, y_min, x_max, y_max])

            # Perform segmentation on the bounding box
            with torch.no_grad():
                sparse_embedding, dense_embedding = self._transform_bbox(self.bbox)
                seg = self._infer(sparse_embedding, dense_embedding)
                torch.cuda.empty_cache()

            self.show_mask(seg)
            self.clear_button.grid(row=1, column=0,pady=10,padx=10)
            self.save_button.grid(row=1, column=1,pady=10,padx=10)
            self.rect = None

            gc.collect()
            
    def _transform_bbox(self, bbox):
        """Transform the bounding box from pixel space to image space.
            Args:
                bbox: 1D NumPy array (x_min, y_min, x_max, y_max)."""
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
        return sparse_embeddings, dense_embeddings

    def _infer(self, sparse_embeddings, dense_embeddings):
        """Perform inference to generate the segmentation mask.
        """
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
        """Display the segmentation mask on the canvas.
            Args:
                mask: 2D NumPy array (Height, Width).
                random_color: Whether to use a random color for the mask.
                alpha: Transparency of the mask (0.0 to 1.0)."""
        
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
        
    def clear(self):
        """Clear the canvas and reset the selections.
            Calls:
                display_image()"""
        self.canvas.delete("all")
        if self.image is not None:
            self.display_image(self.image)

        self.clear_button.grid_remove()
        self.save_button.grid_remove()

    def save(self):
        """Perform segmentation on all slices with bounding box and save the segmentation results
            to a NRRD file.
            Calls:
                get_image_from_plane()
                normalize_to_uint8()
                _set_image()
                _infer()
            """

        slice_range = range(int(self.begin_slice), int(self.end_slice) + 1)
        sparse_embedding, dense_embeddings = self._transform_bbox(self.bbox)
        self.progressbar['value'] = 0
        self.progressbar.grid()

        for idx, slice_index in enumerate(slice_range, 1):           
            current_slice = self.get_image_from_plane(self.plane, slice_index)
            current_slice = self.normalize_to_uint8(current_slice)

            self._set_image(current_slice)

            with torch.no_grad():
                seg = self._infer(sparse_embedding, dense_embeddings)
                torch.cuda.empty_cache()

            self.segs.append(copy.deepcopy(seg))
            gc.collect()
            
             # display the process in the progress bar (takes percentage)
            self.progressbar['value'] = (idx/len(slice_range)) * 100
            self.window.update_idletasks()
        
        segs_array = np.stack(self.segs, axis=0) # segmentations of just selected slices

        # Update the segmented organs list with the new segmentation result
    
        complete_array = self.place_in_3d_array(segs_array)
        
        # If I name organ is not yet in segmented organs list, Update the segmented organs list
        if self.selected_organ not in self.segmented_organs_list:
            self.segmented_organs_list.update({self.selected_organ : [len(self.segmented_organs_list), None]})
            self.segmented_organs_list[self.selected_organ][1] = complete_array
        elif self.selected_organ in self.segmented_organs_list:
            old_array = self.segmented_organs_list[self.selected_organ][1]
            composite_array = old_array + complete_array
            # Only keep the pixels that are 2
            self.segmented_organs_list[self.selected_organ][1] = (composite_array == 2).astype(int)

        
        self.callback(self.segmented_organs_list)
        
        #self.callback(segs_array) #Return segmentation to main UI
        self.progressbar.grid_remove()
        self.window.destroy() # Close pop up window

        # messagebox.showinfo("Saved", "Segmentation result saved to segs.nrrd")
    def place_in_3d_array(self, mask_array):
        dummy_list = np.zeros(self.readdata.shape)
        if self.plane == 0:            
            mask_array = np.transpose(mask_array, (1, 2, 0))
            dummy_list[:, :, int(self.begin_slice) - 1 : int(self.end_slice)] = mask_array
        elif self.plane == 1:
            mask_array = np.transpose(mask_array, (1, 0, 2))
            dummy_list[:, int(self.begin_slice) - 1 : int(self.end_slice), :] = mask_array
        elif self.plane == 2:
            dummy_list[int(self.begin_slice) - 1 : int(self.end_slice), :, :] = mask_array
        return dummy_list

    def get_image_from_plane(self, plane, slice_index):
        """Get the image from a specific plane and slice index.
            Args:
                plane: 0 for axial, 1 for coronal, 2 for sagittal.
                slice_index: Index of the slice."""
        if self.plane == 0:
            image = self.readdata[:, :, slice_index]
        elif self.plane == 1:
            image = self.readdata[:, slice_index, :]
        elif self.plane == 2:
            image = self.readdata[slice_index, :, :]

        return image
    
    def show(self, image, random_color=True, alpha=0.65):
        """Display the image and run the segmentation demo.
            Args:
                image: File path or NumPy array.
                random_color: Boolean whether to use a random color for the mask.
                alpha: Transparency of the mask (0.0 to 1.0)."""
        if isinstance(image, str):
            self.set_image_path(image)
        elif isinstance(image, np.ndarray):
            self._set_image(image)
        else:
            raise ValueError("Input must be a file path or a NumPy array.")
        self.window.mainloop()

    def set_image_path(self, image_path):
        """Load image from file path and set.
            Args:
                image_path: File path to the image."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._set_image(image)

    def _preprocess_image(self, image):
        """Preprocess the image for the model.
            Args:
                image: File path or NumPy array."""
        img_resize = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        return img_tensor

    def on_close(self):
        """Handle the window close event."""
        self.window.destroy()
    
    def on_canvas_resize(self, event):
        """Handle the canvas resize event.
            Args:
                event: Event object."""
        if self.canvas.winfo_width() > 1 and self.canvas.winfo_height() > 1:
            self.display_image(self.image)