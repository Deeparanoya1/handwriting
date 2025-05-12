import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Concatenate
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from scipy.ndimage import sobel

TARGET_SIZE = (256, 256)
DATASET_PATH = r"C:\\FISH\\Writers"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU aktif:", gpus)
else:
    print("⚠️ GPU bulunamadı, CPU kullanılacak.")

def extract_graphological_features(img):
    features = []
    gray = img.squeeze() if img.shape[-1] == 1 else img
    pressure = 1.0 - gray
    features.append(np.mean(pressure))
    features.append(np.std(pressure))
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    angle = np.arctan2(grad_y, grad_x)
    features.append(np.mean(angle))
    features.append(np.std(angle))
    row_sums = np.sum(pressure, axis=1)
    col_sums = np.sum(pressure, axis=0)
    features.append(np.mean(row_sums))
    features.append(np.mean(col_sums))
    contours, _ = cv2.findContours((pressure * 255).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours))
    non_zero_coords = np.column_stack(np.where((pressure * 255).astype('uint8') > 30))
    features.append(np.std(non_zero_coords[:, 0]) if non_zero_coords.shape[0] > 0 else 0.0)
    horiz_proj = np.sum((pressure < 0.05), axis=1)
    pen_lifts = np.sum(horiz_proj > (pressure.shape[1] * 0.9))
    features.append(pen_lifts)
    top_margin = np.max(np.sum(pressure, axis=1))
    bottom_margin = np.min(np.sum(pressure, axis=1))
    features.append(float(top_margin - bottom_margin))
    text_area = np.sum(pressure > 0.3)
    total_area = pressure.shape[0] * pressure.shape[1]
    features.append(text_area / total_area)
    white_cols = np.sum((pressure < 0.05), axis=0)
    features.append(np.sum(white_cols > pressure.shape[0] * 0.9) / pressure.shape[1])
    vertical_projection = np.sum(pressure < 0.8, axis=0)
    gaps = vertical_projection < (pressure.shape[0] * 0.05)
    gap_indices = np.where(gaps)[0]
    features.append(np.mean(np.diff(gap_indices)) if len(gap_indices) > 1 else 0.0)
    contours, _ = cv2.findContours((pressure * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 5]
    features.append(float(np.std(heights)) if len(heights) > 1 else 0.0)
    edges = cv2.Canny((gray * 255).astype('uint8'), 100, 200)
    features.append(np.sum(edges > 0) / edges.size)
    contour_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
    features.append(np.std(contour_areas) if contour_areas else 0.0)
    num_horizontal_lines = np.sum(np.max(gray, axis=1) < 0.95)
    features.append(num_horizontal_lines)
    widths = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 3]
    features.append(np.std(widths) if widths else 0.0)
    ratios = [h / w for c in contours if (w := cv2.boundingRect(c)[2]) > 3 and (h := cv2.boundingRect(c)[3]) > 3]
    features.append(np.mean(ratios) if ratios else 0.0)
    proj = np.sum(pressure > 0.3, axis=1)
    features.append(np.mean(proj))
    blurred = cv2.GaussianBlur((gray * 255).astype('uint8'), (5, 5), 0)
    edges_blur = cv2.Canny(blurred, 100, 200)
    features.append(np.sum(edges_blur > 0) / edges_blur.size)
    features.append(max(heights) / np.mean(heights) if heights else 0.0)
    features.append(max(widths) / np.mean(widths) if widths else 0.0)
    horizontal_proj = np.sum((pressure > 0.2), axis=0)
    mean_val = np.mean(horizontal_proj)
    connection_score = np.sum(horizontal_proj > mean_val * 0.8) / len(horizontal_proj)
    features.append(connection_score)
    connection_regions = np.sum((horizontal_proj > mean_val * 0.8).astype(np.int32))
    features.append(connection_regions)
    coords = np.column_stack(np.where(pressure > 0.2))
    if len(coords) > 10:
        rows, cols = coords[:, 0], coords[:, 1]
        fit = np.polyfit(cols, rows, 1)
        baseline_angle = np.arctan(fit[0])
        residuals = rows - np.polyval(fit, cols)
        features.append(baseline_angle)
        features.append(np.std(residuals))
    else:
        features.extend([0.0, 0.0])
    return np.array(features, dtype='float32')

def generate_pair_list(dataset_path):
    writer_dirs = {}
    for fname in os.listdir(dataset_path):
        if fname.lower().endswith(".png"):
            writer_id = fname.split('_')[0]
            full_path = os.path.join(dataset_path, fname)
            writer_dirs.setdefault(writer_id, []).append(full_path)
    pair_list = []
    writer_ids = list(writer_dirs.keys())
    for writer_id, paths in writer_dirs.items():
        if len(paths) < 2:
            continue
        for i in range(len(paths) - 1):
            pair_list.append((paths[i], paths[i + 1], 1))
            neg_writer = np.random.choice([wid for wid in writer_ids if wid != writer_id])
            neg_path = np.random.choice(writer_dirs[neg_writer])
            pair_list.append((paths[i], neg_path, 0))
    np.random.shuffle(pair_list)
    return pair_list

class SiameseDataGenerator(Sequence):
    def __init__(self, pair_list, batch_size=2, shuffle=True):
        self.pair_list = pair_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.pair_list) // self.batch_size

    def __getitem__(self, index):
        batch = self.pair_list[index * self.batch_size:(index + 1) * self.batch_size]
        img_a, img_b, feat_a, feat_b, labels = [], [], [], [], []
        for (path_a, path_b, label) in batch:
            im_a = self.load_and_preprocess(path_a)
            im_b = self.load_and_preprocess(path_b)
            f_a = extract_graphological_features(im_a)
            f_b = extract_graphological_features(im_b)
            img_a.append(im_a)
            img_b.append(im_b)
            feat_a.append(f_a)
            feat_b.append(f_b)
            labels.append(label)
        return (
            (np.array(img_a), np.array(img_b), np.array(feat_a), np.array(feat_b)),
            np.array(labels)
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pair_list)

    def load_and_preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, TARGET_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

def build_siamese_model(input_shape, feature_shape):
    def build_base_network():
        inp = Input(shape=input_shape)
        x = Conv2D(32, (5,5), activation='relu')(inp)
        x = MaxPooling2D()(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    base_network = build_base_network()

    input_a_img = Input(shape=input_shape)
    input_b_img = Input(shape=input_shape)
    input_a_feat = Input(shape=feature_shape)
    input_b_feat = Input(shape=feature_shape)

    feat_a_img = base_network(input_a_img)
    feat_b_img = base_network(input_b_img)

    merged_a = Concatenate()([feat_a_img, input_a_feat])
    merged_b = Concatenate()([feat_b_img, input_b_feat])

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    distance = Lambda(euclidean_distance)([merged_a, merged_b])
    outputs = Dense(1, activation='sigmoid')(distance)

    return Model([input_a_img, input_b_img, input_a_feat, input_b_feat], outputs)

print("📁 Veri hazırlanıyor...")
pair_list = generate_pair_list(DATASET_PATH)
train_pairs, test_pairs = train_test_split(pair_list, test_size=0.2, random_state=42)

sample_img = cv2.imread(train_pairs[0][0], cv2.IMREAD_GRAYSCALE)
sample_img = cv2.resize(sample_img, TARGET_SIZE)
sample_img = np.expand_dims(sample_img.astype("float32") / 255.0, axis=-1)
sample_feat = extract_graphological_features(sample_img)

train_gen = SiameseDataGenerator(train_pairs, batch_size=2)
test_gen = SiameseDataGenerator(test_pairs, batch_size=2)

model = build_siamese_model(sample_img.shape, sample_feat.shape)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print("🚀 Eğitim başlıyor...")
model.fit(train_gen, validation_data=test_gen, epochs=10)
model.save("el_yazisi_model.h5")
print("✅ Eğitim tamamlandı ve model kaydedildi.")
