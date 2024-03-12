import os
import random
import matplotlib.pyplot as plt
import cv2

class Dataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'images')
        self.mask_folder = os.path.join(root_dir, 'masks')

        self.image_filenames = sorted(os.listdir(self.image_folder))
        self.masks_filenames = sorted(os.listdir(self.image_folder))

        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def plot_random_image(self):
        idx = random.randint(0, len(self) - 1)
        self.plot_image(idx)

    def plot_image(self, idx):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.masks_filenames[idx])

        img = plt.imread(img_path)
        mask = plt.imread(mask_path)


        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[0].axis('off')
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Mask')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    def getWithOriginalImage(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        original = plt.imread(img_path)

        img, mask = self[idx]

        return img, mask, original

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.masks_filenames[idx])

        img = plt.imread(img_path)
        mask = plt.imread(mask_path)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.transform:
            img = self.transform(gray_image)
            mask = self.transform(mask)

        return img, mask
