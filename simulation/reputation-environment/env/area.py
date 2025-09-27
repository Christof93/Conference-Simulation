import pickle

import matplotlib.pyplot as plt
import numpy as np


class Area:
    def __init__(self, xlim=(0, 100), ylim=(0, 100)):
        self.xlim = xlim
        self.ylim = ylim
        self.areas = []  # list of (x0, y0, sigma, value)

    def random_point(self):
        x = np.random.uniform(*self.xlim)
        y = np.random.uniform(*self.ylim)
        return (x, y)

    def random_gaussian_point(self):
        x = np.tanh(
            np.random.normal(
                np.mean(np.array(self.xlim)), 0.3 * np.abs(self.xlim[0] - self.xlim[1])
            )
        )
        y = np.tanh(
            np.random.normal(
                np.mean(np.array(self.ylim)), 0.3 * np.abs(self.ylim[0] - self.ylim[1])
            )
        )
        return (x, y)

    def add_gaussian_area(self, x0, y0, sigma, value):
        """
        Define a Gaussian region centered at (x0,y0) with std=sigma and value scaling.
        """
        self.areas.append((x0, y0, sigma, value))

    @staticmethod
    def distance(p1, p2):
        """
        Compute Euclidean distance between two points p1=(x1,y1) and p2=(x2,y2).
        """
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p2 - p1, axis=1)

    def value_at(self, x, y):
        """
        Return the weighted sum of Gaussian values at a point (x,y),
        plus optional bias plane, squashed with tanh to keep values bounded.
        """
        val = 0
        for x0, y0, sigma, v in self.areas:
            exponent = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)
            val += v * np.exp(exponent)
        return np.tanh(val)

    def save(self, filename):
        """Pickle the Area object to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Unpickle a Area object from a file."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def visualize(self, resolution=200, sampled_points=None, bounds=None):
        """
        Visualize the space as a heatmap.
        bounds = (xmin, xmax, ymin, ymax) to zoom in on a subarea.
        """
        if bounds is None:
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
        else:
            xmin, xmax, ymin, ymax = bounds

        x = np.linspace(xmin, xmax, resolution)
        y = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(x, y)

        # Same calculation for Z as before...
        Z = 0
        for x0, y0, sigma, v in self.areas:
            Z += v * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
        Z = np.tanh(Z)

        plt.figure(figsize=(32, 32))
        plt.imshow(
            Z,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap="bwr",
            vmin=-1.0,
            vmax=1.0,
            alpha=0.8,
        )
        plt.colorbar(label="Value")

        # Plot Gaussian centers that fall inside the bounds
        if self.areas:
            cx, cy = zip(
                *[
                    (x0, y0)
                    for (x0, y0, _, _) in self.areas
                    if xmin <= x0 <= xmax and ymin <= y0 <= ymax
                ]
            )
            if cx:
                plt.scatter(
                    cx,
                    cy,
                    c="yellow",
                    s=80,
                    edgecolors="black",
                    marker="X",
                    label="Gaussian Centers",
                )

        # Plot sampled points if provided
        if sampled_points is not None:
            px, py = zip(
                *[
                    (px, py)
                    for (px, py) in sampled_points
                    if xmin <= px <= xmax and ymin <= py <= ymax
                ]
            )
            if px:
                plt.scatter(
                    px, py, c="black", s=20, edgecolors="white", label="Sampled Points"
                )

        plt.legend()
        plt.title("2D Space (zoomed in)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
