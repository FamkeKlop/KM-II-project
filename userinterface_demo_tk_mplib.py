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
import boxpromptdemo_app


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
        self.segmented_organs_list = dict()
        self.segmented_organs_vars = dict()
        self.colors_list = [(100, 143, 255, 0.7*255), (120, 94, 240, 0.7*255), (220, 38, 127, 0.7*255), (254, 97, 0, 0.7*255), (255, 176, 0, 0.7*255)]

        # Load the MedSAM model
        MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
        device = torch.device('cuda')  # Set to CPU for compatibility
        self.medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        self.medsam_model = self.medsam_model.to(device)
        self.medsam_model.eval()

        # Define plane labels
        self.planes = ['Axial', 'Coronal', 'Sagittal']

        # UI Elements
        self.setup_ui_elements()

    def setup_ui_elements(self):
        """ Set up all GUI components. Set-up divided into 2 frames: main_frame and left_frame """
        # Main Frame for padding and root structure
        self.main_frame = tk.Frame(self.root, padx=20, pady=10)  # Equal padding on both sides
        self.main_frame.grid(sticky='nsew')
        
        # Configure grid weights to allow expansion but keep the Quit button pinned in the bottom right
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)  # No extra weight for the last row with Quit button

        # Left frame for controls and labels
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

        # Label for specifying the plane
        self.plane_label = tk.Label(self.left_frame, text="Specify the plane for slice range", font=('Helvetica', 10))
        self.plane_label.grid(row=5, column=0, pady=(10, 5), sticky="w")
        self.plane_label.grid_remove()

        # Frame for the checkboxes
        self.plane_frame = tk.Frame(self.left_frame)
        self.plane_frame.grid(row=6, column=0, sticky="w", pady=(5, 10))

        # # Checkboxes
        self.plane_var = tk.IntVar(value=0)

        self.axial_checkbox = tk.Radiobutton(self.plane_frame, text="Axial", variable=self.plane_var, value=0,
        command=self.set_plane_for_range)
        self.axial_checkbox.grid(row=0, column=0, padx=(0, 10), sticky="w")
        self.axial_checkbox.grid_remove()

        self.coronal_checkbox = tk.Radiobutton(self.plane_frame, text="Coronal", variable=self.plane_var, value=1,
        command=self.set_plane_for_range)
        self.coronal_checkbox.grid(row=0, column=1, padx=(10, 10), sticky="w")
        self.coronal_checkbox.grid_remove()

        self.sagittal_checkbox = tk.Radiobutton(self.plane_frame, text="Sagittal", variable=self.plane_var, value=2,
        command=self.set_plane_for_range)
        self.sagittal_checkbox.grid(row=0, column=2, padx=(10, 0), sticky="w")
        self.sagittal_checkbox.grid_remove()

        # Frame for "Begin" and "End" integer entries
        self.range_frame = tk.Frame(self.left_frame)
        self.range_frame.grid(row=7, column=0, sticky="w", pady=(5, 10))

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
        self.organ_range_label.grid(row=8, column=0, pady=(5, 5), sticky="w")
        self.organ_range_label.grid_remove()

        # Start Segmentation Button aligned to the bottom left
        self.segmentation_button = tk.Button(self.left_frame, text="Start Segmentation",
                                            command=self.start_segmentation,
                                            bg='#b8cfb9', font=self.button_font, width=18, height=2)
        self.segmentation_button.grid(row=9, column=0, pady=(5, 20), sticky="sw")
        self.segmentation_button.grid_remove()
        
        self.save_segmentations_button = tk.Button(self.left_frame, text="Save Segmentations",
                                            command=self.save_final_segmentations, 
                                            bg='#b8cfb9', font=self.button_font, width=18, height=2)
        self.save_segmentations_button.grid(row=10, column=0, pady=(5, 20), sticky="sw")
        self.save_segmentations_button.grid_remove()
        
        self.label_saved_segmentations = tk.Label(self.left_frame, text="Segmentations saved", font=('Helvetica', 11, 'bold'))
        self.label_saved_segmentations.grid(row=11, column=0, pady=(0, 10), sticky="w")
        self.label_saved_segmentations.grid_remove()

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

        self.update_windowsize()


    def load_image_from_file(self):
        """ Load an image from an NRRD or NIFTI file and display the first slice.
            Calls: 
                update_slider_range(), get_slice() and update_current_plane_label()
        """
        file_path = filedialog.askopenfilename(title="Select NRRD file", filetypes=[("NRRD files", "*.nrrd"), ("NIFTI files", "*.nii.gz")])
        
        if file_path:
            if file_path.endswith('.nrrd'):
                self.readdata, header = nrrd.read(file_path)
            elif file_path.endswith('.nii.gz'):
                nifti_image = nib.load(file_path)
                self.readdata = np.transpose(nifti_image.get_fdata(), (2, 1, 0))
            
            # Extract file name for title and set title
            file_name = file_path.split('/')[-1]
            self.title_label.config(text=f'Image: {file_name}')
            
            self.masks_bool = False
            
            self.update_slider_range()
            slice_image = self.get_slice(self.readdata, self.slice_index)
            self.slice_rgb = self.normalize_to_uint8(slice_image)
            self.display_image(self.slice_rgb, masks_bool = self.masks_bool)
            self.update_current_plane_label()  # Update plane label
            self.title_label.grid()
            self.change_plane_button.grid()
            self.slice_slider.grid()
            self.current_plane_label.grid()
            self.organ_label.grid()
            self.organ_entry.grid()
            self.submit_organ_button.grid()

            #Remove all previous visible widgets
            self.organ_entry.delete(0, tk.END)
            self.selected_organ_label.grid_remove()
            self.organ_range_label.grid_remove()
            self.begin_entry.grid_remove()
            self.begin_label.grid_remove()
            self.end_entry.grid_remove()
            self.end_label.grid_remove()
            self.submit_range_button.grid_remove()
            self.segmentation_button.grid_remove()
            self.label_saved_segmentations.grid_remove()
            self.save_segmentations_button.grid_remove()

            self.update_windowsize()

    def update_windowsize(self):
        """Update the window size such that all the widgets fit."""

        self.root.update_idletasks()
        new_width = self.root.winfo_reqwidth()
        new_height = self.root.winfo_reqheight()
        self.root.geometry(f"{new_width}x{new_height}")


    def update_slider_range(self):
        """ Update the slider range based on the current plane and calls to display the middle slice of the range.
        """
        if self.current_plane == 0:
            self.slice_slider.config(from_=0, to=self.readdata.shape[2] - 1)
        elif self.current_plane == 1:
            self.slice_slider.config(from_=0, to=self.readdata.shape[1] - 1)
        elif self.current_plane == 2:
            self.slice_slider.config(from_=0, to=self.readdata.shape[0] - 1)
        self.slice_index = int(self.slice_slider.cget('to') / 2)  # Set to middle slice
        self.slice_slider.set(self.slice_index)

    def change_plane(self):
        """ Cycles through the planes and updates the slider range and displayed slice.
            Calls: 
                update_slider_range() and get_slice() 
        """
        self.current_plane = (self.current_plane + 1) % len(self.planes)
        self.update_slider_range()
        self.update_slice(self.slice_index)
        self.update_current_plane_label()

    def update_current_plane_label(self):
        """Update the label to display the current plane."""
        self.current_plane_label.config(text=f"Current plane: {self.planes[self.current_plane]}")

    def update_slice(self, index):
        """ Update the displayed slice when the slider is moved.
            Args:
                value: The new value of the slider.
            Calls: 
                get_slice()
        """	
        slice_image = self.get_slice(self.readdata, int(index))
        self.slice_rgb = self.normalize_to_uint8(slice_image)
        # self.display_image(self.slice_rgb, masks_bool = self.masks_bool)
        
        # Update the image display, including masks if enabled
        if any(var.get() for var in self.segmented_organs_vars.values()):
            # If any masks are selected, combine them with the base image
            masks = []
            alpha = 0.7

            # Base image
            current_image = Image.fromarray(self.slice_rgb).convert("RGBA")

            for organ, value in self.segmented_organs_list.items():
                id = value[0]
                if self.segmented_organs_vars[id].get():
                    # Generate a random color for the mask
                    color = self.colors_list[id]

                    # Display the selected masks:
                    slice_mask = self.get_slice(value[1], int(index), mask=True)
                    h, w = slice_mask.shape

                    # Create an RGBA mask
                    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    mask_rgba[..., :3] = np.array(color[:3])
                    mask_rgba[..., 3] = (slice_mask * color[3]).astype(np.uint8)

                    mask_image = Image.fromarray(mask_rgba, mode="RGBA")
                    current_image = Image.alpha_composite(current_image, mask_image)

                    masks.append(organ)

            # Convert the composite image to a format suitable for Tkinter
            self.current_image = ImageTk.PhotoImage(current_image)

            # Display the updated image on the canvas
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
            self.canvas.image = self.current_image

        else:
            # If no masks are selected, just display the base slice
            self.display_image(self.slice_rgb, masks_bool=self.masks_bool)

    def get_slice(self, data, slice_index, mask = False):
        """ Get image from slice index and plane
            Args:
                slice_index: The index of the slice to display.
            Calls:
                normalize_to_uint8(),
                display_image()
        """
        if self.current_plane == 0:
            slice_image = data[:, :, slice_index]
        elif self.current_plane == 1:
            slice_image = data[:, slice_index, :]
        elif self.current_plane == 2:
            slice_image = data[slice_index, :, :]

        if not mask:
            if len(slice_image.shape) == 2:
                slice_image = np.stack([slice_image] * 3, axis=-1)
            
        return slice_image

    def normalize_to_uint8(self, image):
        """Normalize a NumPy array (float image) to uint8.
            Args:
                image: The image to normalize.
            Returns:
                The normalized image (range 0-255)
        """
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_image = np.uint8(normalized_image)
        return normalized_image

    def display_image(self, rgb_image, masks_bool, pil_image = None):
        """Display the loaded RGB image on the Tkinter canvas.
            Args:
                rgb_image: The (preferably normalized) RGB image to display.
        """
        # If not masks are generated or selected just display the image
        if not masks_bool:
            pil_image = Image.fromarray(rgb_image)
            self.current_image = ImageTk.PhotoImage(pil_image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
            self.canvas.image = self.current_image
        # If masks are selected:
        # elif masks_bool:
            # self.current_image = ImageTk.PhotoImage(pil_image)
            

    def set_organ(self):
        """ Update the selected organ label with the value from the organ_entry and displayes it
        on the canvas. Also displays the range options when organ has been selected     
        """

        self.selected_organ = self.organ_entry.get()
        print("Current organ is: ", self.selected_organ)

        # Check if organ prompt is empty
        if not self.selected_organ.strip():
            self.selected_organ_label.config(text = 'Warning: No organ has been defined', foreground = 'red')
            self.selected_organ_label.grid()
            
            # Remove all range-related elements if no organ is selected (anymore)
            self.organ_range_label.grid_remove()
            self.begin_entry.grid_remove()
            self.begin_label.grid_remove()
            self.end_entry.grid_remove()
            self.end_label.grid_remove()
            self.submit_range_button.grid_remove()
            self.segmentation_button.grid_remove()
            self.plane_label.grid_remove()
            self.axial_checkbox.grid_remove()
            self.coronal_checkbox.grid_remove()
            self.sagittal_checkbox.grid_remove()
            self.label_saved_segmentations.grid_remove()
            self.save_segmentations_button.grid_remove()
            return
        
        # if organ is selected, update label and display new prompts
        if self.selected_organ in self.segmented_organs_list.keys():
            new_text = 'Selected organ is ' + self.selected_organ + '\n  existing segmentation will be updated'
        else:
            new_text = 'Selected organ is ' + self.selected_organ
        self.selected_organ_label.config(text=new_text, foreground = 'black', font=('Helvetica', 11,'bold'))
        self.selected_organ_label.grid()    
        self.plane_label.grid()
        self.axial_checkbox.grid()
        self.coronal_checkbox.grid()
        self.sagittal_checkbox.grid()

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
        
        # Remove save data buttons
        self.label_saved_segmentations.grid_remove()
        self.save_segmentations_button.grid_remove()

    def set_plane_for_range(self):
        """Set the changed plane, where the organ range will be specified for."""

        self.current_plane = self.plane_var.get()
        self.update_slider_range()
        self.update_slice(self.slice_index)
        self.update_current_plane_label()

        # Remove previous information
        self.begin_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)
        self.begin_slice = None
        self.end_slice = None
        self.organ_range_label.grid_remove()
        self.segmentation_button.grid_remove()


    def set_organ_range(self):
        """ Update the range label with the values from the begin and end entry boxes and displays
        them on the canvas.
        """
        # Get the values from the entry boxes
        self.begin_slice = self.begin_entry.get()
        self.end_slice = self.end_entry.get()

        #Apply some debugging
        if not self.begin_slice.strip() or not self.end_slice.strip():
            self.organ_range_label.config(text = 'Warning: No slice range is defined', foreground = 'red')
            self.organ_range_label.grid()
            self.segmentation_button.grid_remove()
            self.label_saved_segmentations.grid_remove()
            self.save_segmentations_button.grid_remove()
            return
        
        if not self.begin_slice.isdigit() or not self.end_slice.isdigit():
            self.organ_range_label.config(text = 'Warning: Slice should be an integer', foreground = 'red')
            self.organ_range_label.grid()
            self.segmentation_button.grid_remove()
            self.label_saved_segmentations.grid_remove()
            self.save_segmentations_button.grid_remove()
            return
        
        if int(self.begin_slice) > int(self.end_slice):
            self.organ_range_label.config(text = 'Warning: Begin slice > End slice', foreground = 'red')
            self.organ_range_label.grid()
            self.segmentation_button.grid_remove()
            self.label_saved_segmentations.grid_remove()
            self.save_segmentations_button.grid_remove()
            return

        # Update label/text and display segmentation button
        new_text = "Slice range of " + self.selected_organ + ' is [' + self.begin_slice + ', ' + self.end_slice + ']'
        self.organ_range_label.config(text=new_text, foreground = 'black')
        self.organ_range_label.grid()
        self.segmentation_button.grid()

        # Set displayed image to middle slice of range
        middle_slice = int((float(self.end_slice) - float(self.begin_slice))/2 + float(self.begin_slice))
        self.slice_index = middle_slice
        self.slice_slider.set(self.slice_index)
        self.update_slice(self.slice_index)
 
    def start_segmentation(self):
        """ Start segmentation with the values from the begin and end entry boxes.
        """
        self.save_segmentations_button["state"] = "active"
            
        # Call new tkinter view
        bbox_prompt_demo = boxpromptdemo_app.BboxPromptDemo(self.medsam_model, self.readdata, self.begin_slice, self.end_slice, 
        self.current_plane, self.slice_index, self.root, self.save_segmentation, self.segmented_organs_list, self.selected_organ,)
        bbox_prompt_demo.show(self.slice_rgb)
    
    def choice(self):
        masks = []
        alpha = 0.7
        
        # Base image
        current_image = Image.fromarray(self.slice_rgb).convert("RGBA")
        
        # Loop over all segmented organs
        for organ, value in self.segmented_organs_list.items():
            # If the checkbox is checked, add the organ to the list
            id = value[0]
            
            if self.segmented_organs_vars[id].get():
                color = self.colors_list[id]   
                
                # Display the selected masks:
                slice_mask = self.get_slice(value[1], int(self.slice_index), mask = True)
                h, w = slice_mask.shape
                
                # Create an RGBA mask
                mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                mask_rgba[..., :3] = np.array(color[:3])
                mask_rgba[..., 3] = (slice_mask * color[3]).astype(np.uint8)
                
                mask_image = Image.fromarray(mask_rgba, mode="RGBA")
                current_image = Image.alpha_composite(current_image, mask_image)
                
                masks.append(organ)
            
        # Convert to a format suitable for Tkinter 
        self.current_image = ImageTk.PhotoImage(current_image)
        
        # Display the updated image on the canvas
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
        self.canvas.image = self.current_image
                
        #display_image(self.slice_rgb, masks_bool = self.masks_bool, pil_image)
        
        
    def save_final_segmentations(self):
        # Get most recent segmentation
        final_segmentations = np.zeros(self.readdata.shape)
        for organ, value in self.segmented_organs_list.items():
            number_id = value[0]
            mask_array = value[1]
            final_segmentations += mask_array*(number_id+1)
            
        self.label_saved_segmentations.grid()
        self.save_segmentations_button["state"] = "disabled"
        
        nrrd.write('segmentations.nrrd', final_segmentations)

    def save_segmentation(self, seg):     
        # Right frame for segmented organs section
        self.segmented_organs_frame = tk.Frame(self.main_frame, padx=10)
        self.segmented_organs_frame.grid(row=0, column=2, sticky='n', padx=(0, 20), pady=(10, 10))

        # Label for "Segmented organs"
        self.segmented_organs_label = tk.Label(self.segmented_organs_frame, text="Segmented organs", font=('Helvetica', 11, 'bold'))
        self.segmented_organs_label.grid()
        
        self.save_segmentations_button.grid()
        
        #Remove all previous visible widgets
        self.organ_entry.delete(0, tk.END)
        self.organ_range_label.grid_remove()
        self.selected_organ_label.grid_remove()
        self.begin_entry.grid_remove()
        self.begin_label.grid_remove()
        self.end_entry.grid_remove()
        self.end_label.grid_remove()
        self.submit_range_button.grid_remove()
        self.segmentation_button.grid_remove()
        self.plane_label.grid_remove()
        self.axial_checkbox.grid_remove()
        self.coronal_checkbox.grid_remove()
        self.sagittal_checkbox.grid_remove()
        self.label_saved_segmentations.grid_remove()
        self.save_segmentations_button.grid_remove()
    

        # Checkbox for segmented organs
        for organ, value in self.segmented_organs_list.items():
            # Create BooleanVar for each checkbox
            self.segmented_organs_vars[value[0]] = tk.BooleanVar()  # Variable to track the checkbox state
            segmented_organs_checkbox = tk.Checkbutton(self.segmented_organs_frame, text=organ, 
                                                        variable=self.segmented_organs_vars[value[0]], font=('Helvetica', 10), command=self.choice)
            segmented_organs_checkbox.grid(row=value[0]+1, column=0, sticky="w")

        self.update_windowsize()    #Update windowsize
        
        

# Main section to start the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()