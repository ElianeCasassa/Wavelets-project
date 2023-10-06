from mother_class import Mother_class


class Standard_Wavelet_transform(Mother_class):
    # Implementation of Standard wavelet translation along 2 directions
    def __init__(self, img_file, psi, scale, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method="threshold"):

        folder = "Results_standard_WT/"

        # Directions to take into account in this case
        self.directions = [[1, 0], [0, 1]]

        # Init Mother class
        super().__init__(img_file, folder, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method)

        # Scale and wavelet to use
        self.scale = scale
        self.psi = psi

        # Computation of the gradient magnitude
        self.fi, self.magnitude = self.compute_gradient_magnitude()
        self.plot(self.magnitude, folder, "gradient_magnitude_image")

        # Non-maximum suppression
        self.local_max, self.local_gradient_direction = self.non_maximum_suppression()
        self.plot(self.local_max, folder, "local_maximum_image")

        # Double thresholding of the Mother class
        self.threshold()
