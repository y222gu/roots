import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import os
import torch
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
        return pil_image

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
            
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Sobel Edge Detection (X and Y direction)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        edges_sobel = cv2.sqrt(cv2.add(cv2.pow(sobelx, 2), cv2.pow(sobely, 2)))

        # Normalize the Sobel output to the range [0, 255]
        sobel_normalized = cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the normalized image to uint8 for proper display
        sobel_normalized = np.uint8(sobel_normalized)

        # adjust the brightness of the image
        sobel_normalized = cv2.convertScaleAbs(sobel_normalized, alpha=3, beta=0)

        # Optionally, you can apply a manual threshold to keep only the brightest edges
        edges_canny = cv2.threshold(sobel_normalized, 50, 255, cv2.THRESH_BINARY)[1]


        plt.imsave(os.path.join(os.getcwd(), "sobel_edge_detection.png"), sobel_normalized, cmap='gray')
        return edges_sobel

if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(),"SQR6b.png")

    image = cv2.imread(image_path)
    edge_detector = EdgeDetector(image)
    edge_detector.find_edges()
    # pil_image = edge_detector.auto_mask_generate()