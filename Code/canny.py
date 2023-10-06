import numpy as np
from mother_class import Mother_class


class Canny(Mother_class):
    # Implementation of Canny's algorithm

    def __init__(self, img_file, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method="threshold"):

        folder = "Results_Canny/"

        # Init Mother class and smooth of the image
        super().__init__(img_file, folder, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method)

        # Computation of the gradient magnitude and plot
        self.fi, self.magnitude = self.compute_gradient_magnitude()
        self.plot(self.magnitude, folder, "gradient_magnitude_image")

        # Non-maximum suppression and plot
        self.local_max, self.local_gradient_direction = self.non_maximum_suppression()
        self.plot(self.local_max, folder, "local_maximum_image")

        # Double thresholding of the Mother class
        self.threshold()

    def compute_gradient_magnitude(self):
        # Rewriting of this function in the Canny case
        # The two directions are horizontal and vertical ones
        # We use finite difference scheme
        fi = np.zeros((self.shape_x, self.shape_y, 2))
        magnitude = np.zeros((self.shape_x, self.shape_y))
        for x in range(0, self.shape_x - 1):
            for y in range(0, self.shape_y - 1):
                coeff_x = self.img[x + 1, y] - self.img[x - 1, y]
                coeff_y = self.img[x, y + 1] - self.img[x, y - 1]
                fi[x, y] = [coeff_x, coeff_y]
                magnitude[x, y] = np.sqrt(coeff_x**2 + coeff_y**2)
        return fi, magnitude
