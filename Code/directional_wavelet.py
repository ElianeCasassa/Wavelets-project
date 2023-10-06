import numpy as np
from mother_class import Mother_class


class Directional_wavelet(Mother_class):
    # Implementation of the paper, wavelet translation along 4 directions

    def __init__(self, img_file, psi, scale, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method="threshold"):

        folder = "Results_directionnal/"

        # Directions to take into account in this case
        self.directions = [[1, 0], [1, 1], [0, 1], [-1, 1]]

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

    def first_case_non_maximum(self, x, y, fi_and_dir):
        # Rewriting of the non_maximum_suppression algorithm
        # First case of the algo : they are neighbours

        # first and second directions
        f1, dir1 = fi_and_dir[0]
        f2, dir2 = fi_and_dir[1]
        vect_f1 = f1 * np.array(dir1)
        vect_f2 = f2 * np.array(dir1)
        f1, f2 = np.abs(f1), np.abs(f2)

        # Interpolation of the magnitude of the neighbours pixels along the
        # direction f1 + f2
        MP1 = (0.5 * f2/f1)*self.magnitude[x + dir2[0], y + dir2[1]] + (1 - 0.5 * f2/f1)*self.magnitude[x + dir1[0], y + dir1[1]]
        MP2 = (0.5 * f2/f1)*self.magnitude[x - dir2[0], y - dir2[1]] + (1 - 0.5 * f2/f1)*self.magnitude[x - dir1[0], y - dir1[1]]
        magn_x_y = self.magnitude[x, y]

        # Pixel retain if it is a local maxima
        if magn_x_y > MP1 and magn_x_y > MP2:
            return vect_f1 + vect_f2
        return None

    def second_case_non_maximum(self, x, y, fi, fi_and_dir):
        # Rewriting of the non_maximum_suppression algorithm
        # Second case of the algo : they are not neighbours
        abs_fi = np.abs(fi)
        average = sum(abs_fi) / len(self.directions)
        # Algorithm testes to the first and the second gradient direction
        # The first that satisfy every condition is retain
        for i in range(2):
            if abs_fi[i] > average:
                dir = fi_and_dir[i][1]
                M1 = self.magnitude[x + dir[0], y + dir[1]]
                M2 = self.magnitude[x - dir[0], y - dir[1]]
                magn_x_y = self.magnitude[x, y]
                if magn_x_y > M1 and magn_x_y > M2:
                    perpendicular_vector1 = [- dir[1], dir[0]]
                    perpendicular_vector2 = [dir[1], - dir[0]]
                    perpendicular_f = None
                    for fj, dirj in fi_and_dir:
                        if dirj == perpendicular_vector1 or dirj == perpendicular_vector2:
                            perpendicular_f = fj
                            break
                    if np.abs(perpendicular_f) < average:
                        return dir
        return None

    def non_maximum_suppression(self):
        # Rewriting of the non_maximum_suppression algorithm
        local_max = np.zeros((self.shape_x, self.shape_y))
        local_gradient_direction = np.zeros((self.shape_x, self.shape_y, 2))
        for x in range(1, self.shape_x - 1):
            for y in range(1, self.shape_y - 1):
                fi = self.fi[x, y]
                fi_and_dir = [(fi[i], self.directions[i]) for i in range(len(self.directions))]
                # We sort the direction according to the gradient magnitude
                fi_and_dir.sort(key=lambda fi_dir: np.abs(fi_dir[0]), reverse=True)

                f1, dir1 = fi_and_dir[0]
                f2, dir2 = fi_and_dir[1]
                vect_f1 = f1 * np.array(dir1)
                vect_f2 = f2 * np.array(dir1)

                # Computation of the scalar product between the two main directions
                # They are neighbours if the scalar product is strictly positive
                scalar_product = vect_f1[0]*vect_f2[0] + vect_f1[1]*vect_f2[1]

                if scalar_product > 0:  # They are neighbours
                    retain = self.first_case_non_maximum(x, y, fi_and_dir)
                else:  # They are not neighbours
                    retain = self.second_case_non_maximum(x, y, fi, fi_and_dir)
                if retain is not None:
                    local_max[x, y] = 1
                    local_gradient_direction[x, y] = retain
        return local_max, local_gradient_direction
