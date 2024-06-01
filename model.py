import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import numpy as np
import zipfile

def load_and_preprocess_dataset(zip_file_path, target_size=(128, 128), extraction_folder='/content/extracted_folder'):
    # Create the extraction folder if it doesn't exist
    os.makedirs(extraction_folder, exist_ok=True)

    # Extract the contents of the zip file to the specified folder
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder)
    except Exception as e:
        raise ValueError(f"Error extracting zip file: {e}")

    images = []
    labels = []

    for root, _, files in os.walk(extraction_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(root, filename)
                try:
                    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
                    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize pixel values
                    images.append(image)
                    labels.append(os.path.basename(root))  # Use the parent directory as the label
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    return np.array(images), np.array(labels)

# Load and preprocess dataset from a zip file
zip_file_path = "/content/drive/MyDrive/braintumor.zip"
images, labels = load_and_preprocess_dataset(zip_file_path)

# Ensure the labels are in the expected format
expected_labels = {'glioma', 'meningioma', 'notumor', 'pituitary'}
if not set(labels).issubset(expected_labels):
    raise ValueError("Unexpected labels found in the dataset")

# Create a label mapping
label_mapping = {label: index for index, label in enumerate(expected_labels)}

# Convert labels to numerical format
numerical_labels = np.array([label_mapping[label] for label in labels])

# Split the dataset into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, numerical_labels, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(label_mapping), activation='softmax'))  # Output layer adjusted to the number of classes

# Use sparse categorical crossentropy as the loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('/content/brain_tumor_model.h5')
