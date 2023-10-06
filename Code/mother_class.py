import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class Mother_class:
    # This mother class is usefull for the three algorithms, it allows to use the same functions
    def __init__(self, img_file, record_folder, size_kernel, deviation, outline, val_out, threshold_1, threshold_2, method="threshold"):
        # Load of the image with openCV
        self.img = cv.imread(img_file, 0)

        # Transforms image values between 0 and 1 and in numpy array
        self.img = np.asarray(self.img, dtype="float64")
        self.img /= 255

        # Storage of the parameters
        self.shape_x, self.shape_y = self.img.shape
        self.tau_1 = threshold_1
        self.tau_2 = threshold_2
        self.method = method # method to use for threshold
        self.folder = record_folder

        # Smooth of the image
        self.img = self.Gaussian_smoothing(size_kernel, deviation, outline, val_out)
        # Show the smoothing
        self.plot(self.img, self.folder, "blurred_image")

    def gaussian_filter(self, size_kernel, dev):
        # Return the gaussian kernel of size "size kernel" and with deviation "dev"
        gauss = np.zeros((size_kernel, size_kernel))
        for i in range(size_kernel):
            for j in range(size_kernel):
                x = i - size_kernel // 2
                y = j - size_kernel // 2
                gauss[i, j] = np.exp(-(x**2 + y**2) / (2 * dev**2)) / (2 * np.pi * dev**2)
        return gauss

    def Gaussian_smoothing(self, size_kernel, deviation, outline, val_out):
        # Return the image smoothed

        # We create a new matrix of same size of the image for the output
        smooth_image = np.zeros((self.shape_x, self.shape_y))
        # We compute the gaussian kernel to use
        B = self.gaussian_filter(size_kernel, deviation)

        for i in range(self.shape_x):
            for j in range(self.shape_y):
                # For each pixel we convolve the image with the Gaussian kernel
                sum = 0
                for k in range(- size_kernel // 2, size_kernel // 2 + 1):
                    for l in range(- size_kernel // 2, size_kernel // 2 + 1):
                        value_of_B = B[k + size_kernel // 2, l + size_kernel // 2]
                        if (i - k >= 0) and (i - k < self.shape_x) and (j - l >= 0) and (j - l < self.shape_y):
                            sum += self.img[i - k, j - l] * value_of_B
                        else:
                            # Different methods that we can use to do the convolution on the edges
                            if (outline == "cst"):
                                sum += val_out * value_of_B
                            elif (outline == "periodic"):
                                sum += self.img[(i - k) % self.shape_x, (j - l) % self.shape_y] * value_of_B
                            elif (outline == "repeat_pixel"):
                                sum += self.img[i, j] * value_of_B
                smooth_image[i, j] = sum
        return smooth_image

    def direction(self, x, y, dir):
        # Return the data of the image in a direction "dir" seen as 1D data
        # And the index of the current pixel (x, y) in the data array
        if dir == [1, 0]:
            k = np.arange(self.shape_x)
            res = [self.img[ki, y] for ki in k]
            return res, x
        if dir == [0, 1]:
            k = np.arange(self.shape_y)
            res = [self.img[x, ki] for ki in k]
            return res, y
        if dir == [1, 1]:
            k = np.arange(- min(x, y), min(self.shape_x - x, self.shape_y - y))
            res = [self.img[x + ki, y + ki] for ki in k]
            return res, min(x, y)
        if dir == [-1, 1]:
            k = np.arange(- min(self.shape_x - 1 - x, y), min(x, self.shape_y - 1 - y) + 1)
            res = [self.img[x - ki, y + ki] for ki in k]
            return res, min(self.shape_x - x, y)

    def compute_fi_in_x_y(self, x, y):
        # Compute and return the convolution between the chosen wavelet and
        # every direction in the pixel (x, y)
        fi = np.zeros(len(self.directions))
        for i, dir in enumerate(self.directions):
            # Data of the image in the direction dir and position of the pixel
            f, b = self.direction(x, y, dir)

            # Total version
            # for along_i in range(len(f)):
            #     fi[i] += f[along_i] * self.psi((along_i - b)/self.scale) / np.sqrt(self.scale)

            # Partial version to reduce the cost
            for along_i in range(int(max(b - 4 * self.scale, 0)), int(min(b + 4 * self.scale, len(f)))):
                fi[i] += f[along_i] * self.psi((along_i - b)/self.scale) / np.sqrt(self.scale)
        return fi

    def compute_gradient_magnitude(self):
        # Compute and return the magnitude of the gradient in each point
        # of the image and store in memory the result of the convolution
        # along the different directions
        fi = np.zeros((self.shape_x, self.shape_y, len(self.directions)))
        magnitude = np.zeros((self.shape_x, self.shape_y))
        for x in range(self.shape_x):
            for y in range(self.shape_y):
                fi_in_x_y = self.compute_fi_in_x_y(x, y)
                fi[x, y] = fi_in_x_y
                magnitude[x, y] = np.sqrt(sum(fi_in_x_y**2))
        return fi, magnitude

    def non_maximum_suppression(self):
        # Basic function of non-maximum suppression
        local_max = np.zeros((self.shape_x, self.shape_y))
        local_gradient_direction = np.zeros((self.shape_x, self.shape_y, 2))
        for x in range(1, self.shape_x - 1):
            for y in range(1, self.shape_y - 1):
                fx, fy = self.fi[x, y]

                # Direction of the gradient
                dir = fx * np.array([1, 0]) + fy * np.array([0, 1])

                # Angle associated
                if fx == 0:
                    theta = np.sign(fy) * (np.pi / 2)
                else:
                    theta = np.arctan(fy / fx)

                retain = 0
                # We look if it is a local maxima by comparing on the vertical
                # or the horizontal axe depending on theta
                if - np.pi / 4 < theta <= np.pi / 4:
                    if self.magnitude[x, y] > self.magnitude[x - 1, y] and \
                       self.magnitude[x, y] > self.magnitude[x + 1, y]:
                        retain = 1
                else:
                    if self.magnitude[x, y] > self.magnitude[x, y - 1] and \
                       self.magnitude[x, y] > self.magnitude[x, y + 1]:
                        retain = 1

                if retain:
                    local_max[x, y] = 1
                    local_gradient_direction[x, y] = dir
        return local_max, local_gradient_direction

    def first_threshold(self):
        # First passage of the thresholds, we store in memory the pixels that
        # have a magnitude higher than tau_1 and we store the candidates
        # that are between tau_1 and tau_2
        first_threshold = np.zeros((self.shape_x, self.shape_y))
        candidates = []
        for x in range(self.shape_x):
            for y in range(self.shape_y):
                if self.local_max[x, y]:
                    if self.magnitude[x, y] > self.tau_1:
                        first_threshold[x, y] = 1
                    elif self.tau_2 <= self.magnitude[x, y] <= self.tau_1:
                        candidates.append([x, y])
        return first_threshold, candidates

    def test_between_threshold(self):
        # Function used for plot the pixels that are between tau_1 and tau_2
        threshold = np.zeros((self.shape_x, self.shape_y))
        for x in range(self.shape_x):
            for y in range(self.shape_y):
                if self.local_max[x, y]:
                    if self.tau_2 <= self.magnitude[x, y] <= self.tau_1:
                        threshold[x, y] = 1
        return threshold

    def add_candidates(self):
        # We runs through the set of candidates pixels to see if they are
        # neighbours to an edge until that at one iteratation no pixel is integrated
        add = 1
        while add != 0:
            add = 0
            i = 0
            while i != len(self.candidates): # Runs through the set of candidates
                x, y = self.candidates[i]
                gradient_dir = self.local_gradient_direction[x, y]
                perpendicular_dir = [- int(gradient_dir[1]), int(gradient_dir[0])]
                if self.first_threshold[x + perpendicular_dir[0], y + perpendicular_dir[1]] or \
                    self.first_threshold[x - perpendicular_dir[0], y - perpendicular_dir[1]]:
                    self.first_threshold[x, y] = 1
                    add += 1
                    i -= 1
                    self.candidates.pop(i)
                i += 1
        return self.first_threshold

    def threshold(self):
        # Double threshold scheme
        if self.method == "percent":
            # If the method to use was percent we need to select the good
            # threshold to use
            sort = np.sort(self.magnitude, axis=None)
            shape = len(sort)
            self.tau_1 = sort[int((100 - self.tau_1) * shape / 100)]
            self.tau_2 = sort[int((100 - self.tau_2) * shape / 100)]

        # Firstly we retain in memory the edges and the candidates pixels
        self.first_threshold, self.candidates = self.first_threshold()
        self.plot(self.first_threshold, self.folder, "first_threshold")

        # We plot what is visually the pixels that are between tau_1 and tau_2
        test_between = self.test_between_threshold()
        self.plot(test_between, self.folder, "between_threshold")

        # We verify if the candidates are close to an edge
        result = self.add_candidates()
        self.plot(result, self.folder, "final_result")

    def plot(self, mat, folder, name):
        # Plot the matrix mat given in argument
        plt.imshow(255 * mat, cmap='gray')
        plt.savefig(folder + name)
