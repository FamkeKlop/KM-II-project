import cv2
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import torch
from segment_anything import sam_model_registry
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
from PIL import Image, ImageTk
import nrrd
sys.path.append('functions/')
from utils.demo import BboxPromptDemo


class App:
    def __init__(self, root, window_name="GUI for Segmentation"):
        self.root = root
        self.root.title(window_name)
        self.root.geometry('800x500')
        self.image = None  # To store the loaded image
        self.readdata = None  # To store the loaded NRRD data
        self.slice_index = 0  # Initialize slice index
        self.current_plane = 0  # Initialize current plane index (0: axial, 1: coronal, 2: sagittal)
        self.button_font = tkFont.Font(family="Helvetica", size=11)
        self.slice_rgb = []

        # Load the medSAM model in
        MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
        device = torch.device('cpu')        #Changed it to CPU to work on our laptops
        self.medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        self.medsam_model = self.medsam_model.to(device)
        self.medsam_model.eval()

        # Define the plane labels
        self.planes = ['Axial', 'Coronal', 'Sagittal']

        # Create a frame for the image and the slider
        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas to display the image
        self.canvas = tk.Canvas(self.frame, width=500, height=400)
        self.canvas.pack(fill=tk.X, side = tk.TOP)

        # Slider for selecting the slice index
        self.slice_slider = tk.Scale(self.frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_slice)
        self.slice_slider.pack()
        self.slice_slider.pack_forget()

        # Create a frame for the control buttons and label on the left side
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, padx=20, pady=20, expand = True)

        # Button to load an image
        self.load_button = tk.Button(self.control_frame, text="Load NRRD File", command=self.load_image_from_file, 
        bg = '#abbdd9', font=self.button_font, width=18, height=2)
        self.load_button.pack(side = tk.TOP, pady=20)

        # Button to change planes
        self.change_plane_button = tk.Button(self.control_frame, text="Change Plane", command=self.change_plane, 
        bg = '#f5f3d5', font=self.button_font, width=18, height=2)
        self.change_plane_button.pack(pady=10)
        self.change_plane_button.pack_forget()  # Hide the button initially

        # Label to display the current plane
        self.current_plane_label = tk.Label(self.control_frame,
        font=tkFont.Font(family="Helvetica", size=10, slant='italic'))
        self.current_plane_label.pack(pady=20)
        self.current_plane_label.pack_forget()  # Hide the label initially

        # Button to start segmentation
        self.segmentation_button = tk.Button(self.control_frame, text="Start Segmentation", command=self.start_segmentation,
        bg='#b8cfb9', font=self.button_font, width=18, height=2)
        self.segmentation_button.pack()
        self.segmentation_button.pack_forget()

        # Button to quit the app
        self.quit_button = tk.Button(self.control_frame, text="Quit", command=root.quit, 
        bg = '#3b3a3a', fg='#ffffff', font=self.button_font, width=18, height=2)
        self.quit_button.pack(side=tk.BOTTOM, pady=20)  # Place the Quit button at the bottom

    def load_image_from_file(self):
        """Load an image from an NRRD file and display the first slice."""
        # File selection dialog
        file_path = filedialog.askopenfilename(title="Select NRRD file", filetypes=[("NRRD files", "*.nrrd")])
        
        if file_path:
            # Load the NRRD file
            self.readdata, header = nrrd.read(file_path)
            # Update the slider range based on the number of slices in the first plane (axial)
            self.update_slider_range()
            self.display_slice(self.slice_index)  # Display the first slice

            # Show the Change Plane button
            self.change_plane_button.pack()  # Make the button visible
            self.current_plane_label.pack()  # Show the current plane label
            self.update_current_plane_label()  # Update the label with the current plane
            self.segmentation_button.pack(pady=20)
            self.slice_slider.pack(fill=tk.X, padx=20)

    def update_slider_range(self):
        """Update the slider range based on the current plane."""
        if self.current_plane == 0:  # Axial
            self.slice_slider.config(from_=0, to=self.readdata.shape[2] - 1)
            self.slice_index = int(self.readdata.shape[2] / 2)  # Set to the middle slice
            self.slice_slider.set(self.slice_index)  # Initialize the slider position
        elif self.current_plane == 1:  # Coronal
            self.slice_slider.config(from_=0, to=self.readdata.shape[1] - 1)
            self.slice_index = int(self.readdata.shape[1] / 2)  
            self.slice_slider.set(self.slice_index) 
        elif self.current_plane == 2:  # Sagittal
            self.slice_slider.config(from_=0, to=self.readdata.shape[0] - 1)
            self.slice_index = int(self.readdata.shape[0] / 2)  
            self.slice_slider.set(self.slice_index) 

    def change_plane(self):
        """Change the current plane and update the slider range and displayed slice."""
        self.current_plane = (self.current_plane + 1) % len(self.planes)  # Cycle through planes
        self.update_slider_range()  # Update the slider range based on the new plane
        self.display_slice(self.slice_index)  # Display the first slice of the new plane
        self.update_current_plane_label()  # Update the label with the current plane

    def update_current_plane_label(self):
        """Update the label to display the current plane."""
        self.current_plane_label.config(text=f"Current plane: {self.planes[self.current_plane]}")

    def update_slice(self, value):
        """Update the displayed slice when the slider is moved."""
        self.slice_index = int(value)  # Get the current slider value
        self.display_slice(self.slice_index)  # Display the selected slice

    def display_slice(self, slice_index):
        """Display the selected slice image on the Tkinter canvas."""
        if self.current_plane == 0:  # Axial
            slice_image = self.readdata[:, :, slice_index]
        elif self.current_plane == 1:  # Coronal
            slice_image = self.readdata[:, slice_index, :]
        elif self.current_plane == 2:  # Sagittal
            slice_image = self.readdata[slice_index, :, :]

        # Convert to RGB if grayscale
        if len(slice_image.shape) == 2:
            slice_image = np.stack([slice_image] * 3, axis=-1)
        
        # Normalize the values to [0, 255] for display
        self.slice_rgb = self.normalize_to_uint8(slice_image)

        # Display the image on the canvas
        self.display_image(self.slice_rgb)

    def normalize_to_uint8(self, image):
        """Normalize a NumPy array (float image) to uint8."""
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_image.astype(np.uint8)

    def display_image(self, rgb_image):
        """Display the loaded RGB image on the Tkinter canvas."""
        # Convert RGB to BGR for OpenCV compatibility, then to PIL format
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(bgr_image)

        # Convert to ImageTk format for Tkinter
        tk_image = ImageTk.PhotoImage(pil_image)

        # Display on the canvas
        self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection

    def start_segmentation(self):
        """Open the segmentation interface for the selected slice."""
        if self.readdata is not None:
            # # Start segmentation in a new window
            # segmentation_window = tk.Toplevel(self.root)
            # segmentation_window.title("Segmentation")
            bbox_demo = BboxPromptDemo(self.medsam_model) 
            slice_rgb = np.clip(self.slice_rgb, 0, 1)
            bbox_demo.show(self.slice_rgb)
        else:
            tk.messagebox.showwarning("Warning", "Please load an NRRD file first.")

# Main section to start the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

