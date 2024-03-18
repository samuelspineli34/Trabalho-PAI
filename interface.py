import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import moments_hu

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processing App")
        self.create_widgets()
        self.text_area = None

    def create_widgets(self):
        self.open_button = tk.Button(self.master, text="Open Image", command=self.open_image)
        self.open_button.pack()

        self.convert_button = tk.Button(self.master, text="Convert to Grayscale", command=self.convert_to_grayscale)
        self.convert_button.pack()

        self.histogram_button = tk.Button(self.master, text="Show Histograms", command=self.show_histograms)
        self.histogram_button.pack()

        self.haralick_button = tk.Button(self.master, text="Compute Haralick Descriptors", command=self.compute_haralick_descriptors)
        self.haralick_button.pack()

        self.hu_button = tk.Button(self.master, text="Compute Hu Moments", command=self.compute_hu_moments)
        self.hu_button.pack()

        self.classify_button = tk.Button(self.master, text="Classify Image", command=self.classify_image)
        self.classify_button.pack()

        self.text_area = tk.Text(self.master, height=20, width=80)
        self.text_area.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.text_area.insert(tk.END, "Image opened successfully!\n")

    def convert_to_grayscale(self):
        if hasattr(self, 'image'):
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Grayscale Image", grayscale_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showwarning("Warning", "No image opened.")

    def show_histograms(self):
        if hasattr(self, 'image'):
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            plt.hist(grayscale_image.ravel(), 256, [0, 256])
            plt.title('Grayscale Histogram')
            plt.show()
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
            plt.title('HSV Histogram')
            plt.show()
        else:
            messagebox.showwarning("Warning", "No image opened.")

    def compute_haralick_descriptors(self):
        if hasattr(self, 'image'):
            gray_image = rgb2gray(self.image)
            glcm = greycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
            properties = ['contrast', 'correlation', 'energy', 'homogeneity']
            descriptors = [greycoprops(glcm, prop).ravel()[0] for prop in properties]
            self.text_area.insert(tk.END, "Haralick Descriptors:\n")
            self.text_area.insert(tk.END, f"Contrast: {descriptors[0]}\nCorrelation: {descriptors[1]}\nEnergy: {descriptors[2]}\nHomogeneity: {descriptors[3]}\n")
        else:
            messagebox.showwarning("Warning", "No image opened.")

    def compute_hu_moments(self):
        if hasattr(self, 'image'):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(gray_image)
            hu_moments = cv2.HuMoments(moments).flatten()
            self.text_area.insert(tk.END, "Hu Moments:\n")
            for i in range(7):
                self.text_area.insert(tk.END, f"Hu Moment {i+1}: {hu_moments[i]}\n")
        else:
            messagebox.showwarning("Warning", "No image opened.")

    def classify_image(self):
        messagebox.showwarning("Warning", "Classification not implemented yet.")

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
