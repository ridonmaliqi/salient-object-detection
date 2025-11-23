import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (128,128)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------
# Load image and mask paths
# ---------------------------
def load_image_paths(dataset_root):
    img_paths = sorted(glob.glob(os.path.join(dataset_root, "images", "*")))
    mask_paths = sorted(glob.glob(os.path.join(dataset_root, "masks", "*")))
    assert len(img_paths) == len(mask_paths), "Images and masks count must match"
    return img_paths, mask_paths

# ---------------------------
# Read & preprocess
# ---------------------------
def read_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def read_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def load_pair(img_path, mask_path):
    return read_image(img_path), read_mask(mask_path)

# ---------------------------
# Data augmentation
# ---------------------------
def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    img = tf.image.random_brightness(img, max_delta=0.2)

    # Random crop & resize (simulates cutout/zoom)
    stacked = tf.concat([img, mask], axis=-1)
    crop_size = tf.random.uniform([], int(IMG_SIZE[0]*0.8), IMG_SIZE[0], dtype=tf.int32)
    stacked = tf.image.random_crop(stacked, size=(crop_size, crop_size, 4))
    stacked = tf.image.resize(stacked, IMG_SIZE)
    img = stacked[..., :3]
    mask = stacked[..., 3:]
    return img, mask

# ---------------------------
# Build TensorFlow Dataset
# ---------------------------
def build_tf_dataset(img_paths, mask_paths, augment_data=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.shuffle(1000)
    ds = ds.map(lambda x, y: load_pair(x, y), num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ---------------------------
# Split datasets
# ---------------------------
def create_datasets(dataset_root):
    img_paths, mask_paths = load_image_paths(dataset_root)

    train_img, test_img, train_mask, test_mask = train_test_split(
        img_paths, mask_paths, test_size=0.15, random_state=42
    )
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_img, train_mask, test_size=0.1765, random_state=42
    )

    train_ds = build_tf_dataset(train_img, train_mask, augment_data=True)
    val_ds = build_tf_dataset(val_img, val_mask, augment_data=False)
    test_ds = build_tf_dataset(test_img, test_mask, augment_data=False)

    return train_ds, val_ds, test_ds

# ---------------------------
# Test
# ---------------------------
if __name__ == "__main__":
    dataset_path = "DUTS"
    train_ds, val_ds, test_ds = create_datasets(dataset_path)
    for imgs, masks in train_ds.take(1):
        print("Images batch shape:", imgs.shape)
        print("Masks batch shape:", masks.shape)
