import numpy as np
import nrrd
import slicerio
import json
import torch
import matplotlib.pyplot as plt


def show_slice(data, axis, i):
    """
    Displays one specific slice
    
    Param data: np array (3D) with all images
    Param axis: along which axis the slice is chosen options = (x, y, z)
    Param i: ith slice to be selected
    """
    if axis == "x":
        plt.imshow(data[i,:,:] ,cmap='gray')
    elif axis == "y":
        plt.imshow(data[:,i,:], cmap='gray')
    elif axis == "z":
        plt.imshow(data[:,:,i], cmap='gray')
    plt.axis("off")
    plt.show()
"""
def list_images(data, axis):
    
    Makes a list of all images along a certain axis
    
    Params data: 3D np array with all image data
    Params axis: Along which axis images should put into the list
        choose between 'x', 'y' or 'z'
    
    Returns images: list with al images along chosen axis
    
    x, y, z = data.shape
    if axis == "x":
        images = data.reshape()
    elif axis == "y":
        images = data.reshape()
    elif axis == "z":
        images = data.reshape()
"""    

# Select example image -> reads image as numpy array
readdata, header = nrrd.read('50kVp_0000.dcm.nrrd')

# Select the segmentation file
segmentation = slicerio.read_segmentation('Segmentation.seg.nrrd')

print("Shape of nrrd file", readdata.shape)
# show_slice(readdata, 'z', 200)

from tkinter import *
from PIL import ImageTk, Image

def forward(img_no):
    global label
    global button_forward
    global button_back
    global button_exit
    label.grid_forget()
    
    image = ImageTk.PhotoImage(image=Image.fromarray(readdata[:,:,img_no - 1]))
    label = Label(image=image)
    label.image = image
    label.grid(row=1, column=0, columnspan=3)
    
    button_forward = Button(root, text="forward",
                        command=lambda: forward(img_no+1))

    if img_no == 596:
        button_forward = Button(root, text="Forward",
                                state=DISABLED)

    button_back = Button(root, text="Back",
                         command=lambda: back(img_no-1))

    button_back.grid(row=5, column=0)
    button_exit.grid(row=5, column=1)
    button_forward.grid(row=5, column=2)

def back(img_no):
    global label
    global button_forward
    global button_back
    global button_exit
    label.grid_forget()
    
    image = ImageTk.PhotoImage(image=Image.fromarray(readdata[:,:,img_no - 1]))
    label = Label(image=image)
    label.image = image
    label.grid(row=1, column=0, columnspan=3)
    button_forward = Button(root, text="forward",
                            command=lambda: forward(img_no + 1))
                            
    button_back = Button(root, text="Back",
                         command=lambda: back(img_no - 1))

    if img_no == 1:
        button_back = Button(root, text="Back", state=DISABLED)

    label.grid(row=1, column=0, columnspan=3)
    button_back.grid(row=5, column=0)
    button_exit.grid(row=5, column=1)
    button_forward.grid(row=5, column=2)

# Calling the Tk (The initial constructor of tkinter)
root = Tk()
root.title("Image Viewer")
root.geometry("700x700")

first_image = ImageTk.PhotoImage(image=Image.fromarray(readdata[:,:,0]))
label = Label(image=first_image)
label.image = first_image
label.grid(row=1, column=0, columnspan=3)

# We will have three button back ,forward and exit
button_back = Button(root, text="Back", command=back,
                     state=DISABLED)

# root.quit for closing the app
button_exit = Button(root, text="Exit",
                     command=root.quit)

button_forward = Button(root, text="Forward",
                        command=lambda: forward(1))

# grid function is for placing the buttons in the frame
button_back.grid(row=5, column=0)
button_exit.grid(row=5, column=1)
button_forward.grid(row=5, column=2)

root.mainloop()











###########################################################

#print(header)
#print(type(readdata))

#number_of_segments = len(segmentation["segments"])
#print(f"Number of segments: {number_of_segments}")

#segment_names = slicerio.segment_names(segmentation)
#print(f"Segment names: {', '.join(segment_names)}")#

#segment0 = slicerio.segment_from_name(segmentation, segment_names[0])
#print("First segment info:\n" + json.dumps(segment0, sort_keys=False, indent=4))

#img = readdata.detach()

# Go through each z
#for z in range(readdata.shape[2]):
    