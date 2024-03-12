from torchvision import transforms
import torch
from PIL import Image
from torchvision.ops import masks_to_boxes
from matplotlib import patches
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

def visualize_masks(original, y_pred:torch.tensor, y:torch.tensor,  output_path:str = 'output.png'):
    resize = transforms.Resize((640, 640), interpolation=Image.NEAREST)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    y_resized = resize(y)
    y_resized.shape
    color = (random.random(), random.random(), random.random())

    axes[0][0].imshow(original, cmap='gray')
    bbox = masks_to_boxes(y_resized)[0]
    patch = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor=color, facecolor=color, alpha=0.4)
    axes[0][0].add_patch(patch)
    patch = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor=color, facecolor='none')
    axes[0][0].add_patch(patch)
    axes[0][0].set_title('Target Mask')
    axes[0][0].axis('off')


    y_pred_resized = resize(y_pred)
    color = (random.random(), random.random(), random.random())

    axes[0][1].imshow(original, cmap='gray')
    bbox = masks_to_boxes(y_pred_resized.squeeze(dim=0))[0]
    patch = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor=color, facecolor=color, alpha=0.4)
    axes[0][1].add_patch(patch)
    patch = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor=color, facecolor='none')
    axes[0][1].add_patch(patch)

    axes[0][1].set_title('Predicted Mask')
    axes[0][1].axis('off')

    axes[1][0].imshow(y.squeeze().numpy(), cmap='gray')
    axes[1][0].axis('off')

    axes[1][1].imshow(y_pred.squeeze().numpy(), cmap='gray')
    axes[1][1].axis('off')

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path)

def visualizeTraining(train_losses, val_losses):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Loss and Validation Loss')
    plt.legend()
    plt.show(block=False)

def display_images(image_paths, raw_annotations_data):
    _, axs =  plt.subplots(2, 2, figsize=(10, 10))

    for ax, img_path in zip(axs.ravel(), image_paths):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')

        img_filename = os.path.basename(img_path)

        img_id = [i['id'] for i in raw_annotations_data['images'] if i['file_name'] == img_filename]
        img_annotations = [a for a in raw_annotations_data['annotations'] if a['image_id'] in img_id]

        for ann in img_annotations:
            for seg in ann['segmentation']:

                color = (random.random(), random.random(), random.random())

                polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                patch = patches.Polygon(polygon, closed=True, edgecolor=color, facecolor=color, alpha=0.4)
                ax.add_patch(patch)
                patch = patches.Polygon(polygon, closed=True, edgecolor=color, facecolor='none')
                ax.add_patch(patch)
