from directional_wavelet import Directional_wavelet
from standard_WT import Standard_Wavelet_transform
from canny import Canny
import numpy as np


def psi(t):
    # Wavelet to use
    return -t * np.exp(-t**2/2)/np.sqrt(2*np.pi)


##### Parameters ######
img_file = "Images/pika.png"
scale = 1
size_kernel = 3
deviation = 1
outline = "repeat_pixel"
val_out = 1
threshold_1 = 30
threshold_2 = 70
method = "percent"
#######################

# Computation of the edges depending of the method use
Directional_wavelet(img_file, psi, scale, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method)
Standard_Wavelet_transform(img_file, psi, scale, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method)
Canny(img_file, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method)
