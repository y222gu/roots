import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import os
import torch
from scipy.signal import find_peaks
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
class EdgeDetector():
    def __init__(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.image = image
        model_path = os.path.join(os.getcwd(), "sam_vit_b_01ec64.pth")
        self.sam = sam_model_registry["vit_b"](checkpoint=model_path)
        if torch.cuda.is_available():
            self.sam.to('cuda')

    def auto_mask_generate(self):
        generator = SamAutomaticMaskGenerator(model=self.sam)
        masks = generator.generate(self.image)

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(288*px, 216*px))
        ax.imshow(self.image)
        self.show_anns(masks, ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # save the figure
        plt.savefig(os.path.join(os.getcwd(), "masked_image.png"))
        plt.close(fig)
        return masks


    def point_prompt_mask_generate(self, x=600, y=600, label=1):
        input_point = np.array([[x, y]])
        input_label = np.array([label])
        generator = SamPredictor(self.sam)
        generator.set_image(self.image)
        masks, scores, logits = generator.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        self.best_mask = masks[0]
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(288*px, 216*px))
        ax.imshow(self.image)
        self.show_mask(self.best_mask, ax)
        self.show_points(input_point, input_label, ax, '*', marker_size=100)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # save the figure
        plt.savefig(os.path.join(os.getcwd(), "masked_image.png"))
        plt.close(fig)
        return masks

    def box_prompt_mask_generate(self, x_low=950, y_low=180, x_high=1400, y_high=620):
        input_box = np.array([x_low, y_low, x_high, y_high])
        generator = SamPredictor(self.sam)
        generator.set_image(self.image)
        masks, scores, logits = generator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(288*px, 216*px))
        ax.imshow(self.image)
        self.show_mask(masks[0], ax)
        self.show_box(input_box, ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # save the figure
        plt.savefig(os.path.join(os.getcwd(), "masked_image.png"))
        plt.close(fig)
        return masks

    @staticmethod
    def show_anns(anns, ax=None):
        if len(anns) == 0:
            return
        if ax is None:
            ax = plt.gca()
            ax.set_autoscale_on(False)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.5)))

    @staticmethod
    def show_points(coords, labels, ax, marker, marker_size=100):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='#f39c12', marker=marker, s=marker_size, linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker=marker, s=marker_size, linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    # a function to find the edges of the image
    def find_edges(self):
                # Assuming 'self.image' contains your image data
        image = self.image[:, :, 1]

        # Normalize the image
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Apply Gaussian blur to smooth the image
        image = cv2.GaussianBlur(image, (21, 21), 0)

        # Find local maxima in the smoothed image
        # Apply a threshold to only consider significant peaks
        threshold_abs = 0.85 * np.max(image)  # adjust threshold for significant peaks

        # Use peak_local_max to get local maxima positions
        local_max = peak_local_max(image, min_distance=20, threshold_abs=threshold_abs)

        # Plot the surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0, image.shape[1], 1)
        y = np.arange(0, image.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, image, cmap='viridis')

        # Highlight the local maxima on the plot
        ax.scatter(local_max[:, 1], local_max[:, 0], image[local_max[:, 0], local_max[:, 1]], color='r', s=50)
        print("the number of local maxima is: ", len(local_max))

        plt.show()

        # Print the (x, y) positions of the significant local maxima
        print("Significant local maxima positions (x, y):")
        for peak in local_max:
            print(f"({peak[1]}, {peak[0]}) - Value: {image[peak[0], peak[1]]}")

        # plot the dots on the original image
        plt.figure()
        plt.imshow(self.image)
        plt.scatter(local_max[:, 1], local_max[:, 0], color='r', s=50)
        plt.show()

        return local_max

    
if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(),"images","Truc_cells.png")

    image = cv2.imread(image_path)
    edge_detector = EdgeDetector(image)
    # edge_detector.find_edges()
    pil_image = edge_detector.auto_mask_generate()