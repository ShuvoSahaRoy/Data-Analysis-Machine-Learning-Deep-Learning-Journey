import cv2
import os
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import math

# Create an instance of the ImageDataGenerator
datagen = ImageDataGenerator(
    brightness_range=(0.5, 1.3),  # Random brightness between 0.5 and 1.5
)

# Set the parent folder path
parent_folder = 'original dataset'
output_folder = 'augmented dataset'  # Folder to save the augmented images

# Define the operations
operations = {
    'original': '',
    'brightness': 'brightness',
    'blur': 'blur',
    'sharpness': 'sharpness',
    'shadow': 'shadow',
    'median': 'median',
    'motion_blur': 'motion_blur',
    'gamma': 'gamma',
}

# Iterate over subfolders within the parent folder
for subfolder in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue  # Skip if the item is not a subfolder

    print(f"Processing subfolder: {subfolder}")

    # Create a subfolder within the output folder to save augmented images
    output_subfolder = os.path.join(output_folder, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)

    # Get the list of image files in the current subfolder
    image_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

    # Select 5 random images from the current subfolder
    selected_images = random.sample(image_files, 5)

    # Generate augmented data for each selected image
    for image_name in selected_images:
        image_path = os.path.join(subfolder_path, image_name)

        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate augmented images
        augmented_images = []
        augmented_images.append(image)  # Save the original image

        # Apply brightness augmentation
        augmented_image = datagen.apply_transform(image.copy(), {'brightness': random.uniform(0.5, 1.5)})
        augmented_images.append(augmented_image)

        # Apply blur augmentation
        augmented_image = cv2.GaussianBlur(image, (5, 5), 0)
        augmented_images.append(augmented_image)

        # Apply sharpness augmentation
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        augmented_image = cv2.filter2D(image, -1, sharpen_kernel)
        augmented_images.append(augmented_image)

        # Apply shadow augmentation
        shadowed_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        shadowed_image = np.array(shadowed_image, dtype=np.float64)
        brightness_coefficient = 0.7  # Controls the darkness of the shadow
        shadowed_image[:, :, 2] = shadowed_image[:, :, 2] * brightness_coefficient
        shadowed_image[:, :, 2][shadowed_image[:, :, 2] > 255] = 255
        shadowed_image = np.array(shadowed_image, dtype=np.uint8)
        shadowed_image = cv2.cvtColor(shadowed_image, cv2.COLOR_HSV2RGB)
        augmented_images.append(shadowed_image)

        # Apply median filtering augmentation
        augmented_image = cv2.medianBlur(image, 5)
        augmented_images.append(augmented_image)

        # Apply motion blur augmentation
        kernel_motion_blur = np.zeros((9, 9))
        kernel_motion_blur[int((9 - 1) / 2), :] = np.ones(9)
        kernel_motion_blur = kernel_motion_blur / 9
        motion_blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)
        augmented_images.append(motion_blurred_image)

        # Apply gamma correction augmentation
        def apply_gamma_correction(image, gamma):
            # Normalize the pixel values between 0 and 1
            image = image.astype(float) / 255.0

            # Apply gamma correction
            image = np.power(image, gamma)

            # Scale the pixel values back to the range of 0-255
            image = image * 255.0
            image = np.clip(image, 0, 255)

            # Convert the pixel values to unsigned 8-bit integers
            image = image.astype(np.uint8)

            return image

        gamma = random.uniform(0.5, 2.0)
        augmented_image = apply_gamma_correction(image.copy(), gamma)
        augmented_images.append(augmented_image)


        # Save the augmented images
        base_name = os.path.splitext(image_name)[0]
        for i, augmented_image in enumerate(augmented_images):
            # operation_name = operations[list(operations.keys())[i]]
            # augmented_image_name = f"{base_name}_{operation_name}.jpg"
            augmented_image_name = f"{base_name}{i}.jpg"
            augmented_image_path = os.path.join(output_subfolder, augmented_image_name)
            cv2.imwrite(augmented_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

        print(f"Generated augmented images for: {image_name}")

print("Augmentation completed!")
