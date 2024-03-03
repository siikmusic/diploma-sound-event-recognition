import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import librosa
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from skimage.transform import resize
import Augumentations
import Models
from audio_globals import n_mels, slice_lenght, overlap, n_mfcc, n_fft, hop_length
from librosa.feature import melspectrogram
import matplotlib.colors as colors
from augumentations.CyclicShiftAugmentation import cyclic_shift_augmentation
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.applications.resnet import preprocess_input
from PIL import Image


def wait_for_file(filepath, timeout=10, check_interval=0.5):
    """
    Wait for a file to become accessible within a timeout period.
    :param filepath: Path to the file.
    :param timeout: Maximum time to wait in seconds.
    :param check_interval: Time interval between checks in seconds.
    :return: True if the file becomes accessible, False if timed out.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_file_accessible(filepath):
            return True
        time.sleep(check_interval)
    return False


def extract_labels_from_filenames(directory):
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            label = int(filename.split('_')[0])
            if label not in labels:
                labels.append(label)
    return labels


def peak_normalize(audio, target_peak=1):
    max_amplitude = np.max(np.abs(audio))
    scaling_factor = target_peak / max_amplitude
    normalized_audio = audio * scaling_factor
    return normalized_audio


def remove_silence(y, top_db=20):
    non_silent_parts = librosa.effects.split(y, top_db=top_db)

    non_silent_audio = np.concatenate([y[start:end] for start, end in non_silent_parts])

    return non_silent_audio


def map_labels(label, label_mapping):
    return label_mapping[label]


def is_file_accessible(filepath, mode='r'):
    """
    Check if a file is accessible.
    :param filepath: Path to the file.
    :param mode: Mode to open the file. Default is 'r' (read mode).
    :return: True if the file is accessible, False otherwise.
    """
    try:
        with open(filepath, mode):
            return True
    except IOError:
        return False


class AudioDataset:

    def __init__(self, csv_file, save_dir, sr=44100, feature_type='melspectogram', augumentations=None, model_name=Models.VGG):
        if augumentations is None:
            augumentations = ['original']
        else:
            augumentations.append('original')
        self.augumentations = augumentations
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataset = None
        self.csv_file = csv_file
        self.save_dir = save_dir
        self.sr = sr
        self.df = None
        self.num_classes = 0
        self.label_one_hot_class = {}
        self.save_dir_train = os.path.join("data", "train")
        self.save_dir_val = os.path.join("data", "val")
        self.save_dir_test = os.path.join("data", "test")
        self.train_files = [os.path.join(self.save_dir_train, f) for f in os.listdir(self.save_dir_train)]
        self.test_files = [os.path.join(self.save_dir_test, f) for f in os.listdir(self.save_dir_test)]
        self.val_files = [os.path.join(self.save_dir_val, f) for f in os.listdir(self.save_dir_val)]
        self._load_csv()
        # TODO: labels from meta
        self.unique_labels = extract_labels_from_filenames(self.save_dir_train)
        self.label_mapping = {original_label: new_label for new_label, original_label in enumerate(self.unique_labels)}
        self.feature_type = feature_type
        self.model = model_name

    def _load_csv(self):
        # Load the entire CSV file
        full_df = pd.read_csv(self.csv_file)

        # Identify 8 unique classes - this assumes 'label' contains numerical class identifiers
        unique_classes = ["door_wood_knock","fireworks","crackling_fire","pouring_water","dog","crying_baby","glass_breaking","siren"]

        # Filter the DataFrame to only include rows with those 8 classes
        self.df = full_df[full_df['category'].isin(unique_classes)]

        # Update the number of classes
        self.num_classes = len(unique_classes)

        # Initialize or clear the dictionary (assuming it exists)
        self.label_one_hot_class = {}

        # Map labels to categories for the 8 classes
        for index, row in self.df.iterrows():
            self.label_one_hot_class[row['label']] = row['category']

    def preprocess_train_data(self):

        def clear_directory(directory):
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            else:
                os.makedirs(directory)

        clear_directory(self.save_dir_train)
        clear_directory(self.save_dir_val)
        clear_directory(self.save_dir_test)
        train_df, test_df = train_test_split(self.df, test_size=0.2, stratify=self.df['label'])
        train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['label'])

        self._process_and_save(train_df, self.save_dir_train)
        print("Train features extracted")
        self._process_and_save(val_df, self.save_dir_val, is_test=True)
        print("Val features extracted")
        self._process_and_save(test_df, self.save_dir_test, is_test=True)
        print("Test features extracted")
        self.train_files = [os.path.join(self.save_dir_train, f) for f in os.listdir(self.save_dir_train)]
        self.test_files = [os.path.join(self.save_dir_test, f) for f in os.listdir(self.save_dir_test)]
        self.val_files = [os.path.join(self.save_dir_val, f) for f in os.listdir(self.save_dir_val)]

    def get_datasets(self, batch_size=32):
        self.train_dataset = self.create_dataset(self.train_files, batch_size)
        self.test_dataset = self.create_dataset(self.test_files, batch_size)
        self.val_dataset = self.create_dataset(self.val_files, batch_size)
        return self.train_dataset, self.val_dataset, self.test_dataset

    def process_test(self, file_path):
        wav, _ = librosa.load(file_path, sr=self.sr)
        overlap_amount = slice_lenght * (overlap / 100)
        step_size = slice_lenght - overlap_amount
        num_slices = int(max(1, 1 + (len(wav) - slice_lenght) // step_size))

        for slice_num in range(num_slices):
            start = int(slice_num * step_size)
            end = int(start + slice_lenght)
            wav_slice = wav[start:end]
            if len(wav_slice) < slice_lenght:
                wav_slice = np.pad(wav_slice, (0, slice_lenght - len(wav_slice)), mode='constant')
            feature = self._extract_feature(wav_slice, self.sr)
            feature = librosa.util.normalize(feature)
            file_name = f'{file_path.split(".")[0]}_{os.path.basename(file_path).split(".")[0]}_{slice_num}.npy'
            np.save(file_name, feature)

    def _process_and_save(self, df, save_dir, is_test=False, model_name=Models.RESNET):
        slice_length_samples = int(slice_lenght * self.sr)
        overlap_feature = overlap
        if self.feature_type == 'spectrogram':
            overlap_feature = 75
        overlap_amount = slice_length_samples * (overlap_feature / 100)
        step_size = slice_length_samples - overlap_amount

        for _, row in df.iterrows():
            file_path = row['path']
            label = row['label']
            wav, _ = librosa.load(file_path, sr=self.sr)

            augmentations = self.augumentations if not is_test else ['original']
            for aug in augmentations:
                augmented_wav = self.apply_augmentation(wav, aug)

                num_slices = int(max(1, 1 + (len(augmented_wav) - slice_length_samples) // step_size))
                for slice_num in range(num_slices):
                    start = int(slice_num * step_size)
                    end = int(start + slice_length_samples)
                    wav_slice = augmented_wav[start:end]

                    if len(wav_slice) < slice_length_samples:
                        wav_slice = np.pad(wav_slice, (0, slice_length_samples - len(wav_slice)), mode='constant')

                    feature = self._extract_feature(wav_slice, self.sr)
                    feature = librosa.util.normalize(feature)

                    base_file_name = os.path.basename(file_path).split('.')[0]
                    image_file_name = f'{base_file_name}_{slice_num}_{aug}.png'
                    image_path = os.path.join(save_dir, image_file_name)

                    # Save the feature as an image
                    plt.imsave(image_path, feature, cmap='hot')

    def get_random_audio_and_feature(self):
        random_row = self.df.iloc[0]

        y, sr = librosa.load("siren.wav", sr=None)

        feature = self._extract_feature(y, sr)
        audio_class = 'siren'
        return y, feature, audio_class, "siren.wav"

    def visualize_random_spectrogram_audio_augumented(self, augumentation=Augumentations.PITCH_SHIFT):
        y, spectrogram, audio_class, path = self.get_random_audio_and_feature()
        self.visualize_spectrogram_audio(spectrogram, y, audio_class)
        y_aug = self.apply_augmentation(y, augumentation)
        spectrogram = self._extract_feature(y, self.sr)
        self.visualize_spectrogram_audio(spectrogram, y_aug, audio_class)
        return y, y_aug

    def visualize_spectrogram_audio(self, spectrogram, y, audio_class, x_axis="time", cmap="plasma"):

        plt.figure(figsize=(15, 6))
        Time = np.linspace(0, len(y) / self.sr, num=len(y))

        # Waveform plot
        plt.subplot(2, 1, 1)
        plt.title('Waveform of class')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title("Waveform of class: " + audio_class)
        plt.plot(Time, y)

        plt.subplot(2, 1, 2)
        if self.feature_type == 'melspectogram':
            y_axis_type = "mel"
        elif self.feature_type == 'spectrogram':
            y_axis_type = "log"
        elif self.feature_type == 'mfcc':
            y_axis_type = "none"
        else:
            raise ValueError("Unsupported feature type. Choose 'mel', 'spectrogram', or 'mfcc'.", self.feature_type)
        # Set the minimum dB level for black color
        vmin = -80

        # Create a Normalize object with vmin set to -80 dB
        norm = colors.Normalize(vmin=vmin, vmax=np.max(spectrogram))
        if self.feature_type == 'mfcc':
            librosa.display.specshow(spectrogram)
        else:
            librosa.display.specshow(spectrogram, sr=self.sr, hop_length=hop_length, x_axis=x_axis, y_axis=y_axis_type,
                                     cmap=cmap, norm=norm)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram of class: ' + audio_class)
        plt.tight_layout()
        plt.show()

    def visualize_spectrogram_melspectrogram(self, spectrogram, y, audio_class, x_axis="time", cmap="plasma"):

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=self.sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        spectrogram = librosa.stft(y)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        librosa.display.specshow(mel_spectrogram_db, sr=self.sr, x_axis='time', y_axis='mel', ax=ax[0])
        ax[1].set_title('Mel Spectrogram of class:' + audio_class)

        librosa.display.specshow(spectrogram_db, sr=self.sr, x_axis='time', y_axis='log', ax=ax[1])
        ax[0].set_title('Regular Spectrogram of class:' + audio_class)

        plt.tight_layout()
        plt.show()

    def get_num_classes(self):
        return self.num_classes

    def get_stft(self, window='hann'):
        random_file = random.choice(self.train_files + self.test_files + self.val_files)

        y, sr = librosa.load(random_file, sr=None)

        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
        stft_db = librosa.amplitude_to_db(np.abs(stft))

        return stft_db

    def create_dataset(self, file_paths, batch_size):
        def spectrogram_generator():
            for file_path in file_paths:
                spectrogram, label = self._load_spectrogram(file_path)
                yield spectrogram, label

        sample_shape = self.get_spectrogram_shape()
        dataset = tf.data.Dataset.from_generator(
            spectrogram_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(sample_shape, (self.num_classes,))
        )
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        return dataset

    def _extract_feature(self, y, sr):
        if self.feature_type == 'melspectogram':
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=4096, hop_length=2048, fmin=300, fmax=15000)
            return librosa.power_to_db(S)
        elif self.feature_type == 'spectrogram':
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024)) ** 2
            return S
        elif self.feature_type == 'mfcc':
            return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        else:
            raise ValueError("Unsupported feature type")

    def apply_augmentation(self, wav_slice, aug):
        if aug == 'pitch_shift':
            pitch_shift = np.random.randint(-10, 10)
            wav_slice = librosa.effects.pitch_shift(wav_slice, sr=self.sr, n_steps=pitch_shift)
        elif aug == 'time_stretch':
            stretch_rate = np.random.uniform(0.6, 1.9)
            wav_slice = librosa.effects.time_stretch(wav_slice, rate=stretch_rate)
        elif aug == 'noise_addition':
            noise_level = np.random.uniform(0.001, 0.01)  # Noise level
            wav_slice = wav_slice + noise_level * np.random.randn(len(wav_slice))
        elif aug == Augumentations.CYCLIC_SHIFT_1:
            wav_slice = cyclic_shift_augmentation(wav_slice, self.sr, frame_length=n_fft, hop_length=hop_length,
                                                  energy_threshold=0.0005,
                                                  shift=int(Augumentations.CYCLIC_SHIFT_1.split("_")[-1]))
        elif aug == Augumentations.CYCLIC_SHIFT_2:
            wav_slice = cyclic_shift_augmentation(wav_slice, self.sr, frame_length=n_fft, hop_length=hop_length,
                                                  energy_threshold=0.0005,
                                                  shift=int(Augumentations.CYCLIC_SHIFT_2.split("_")[-1]))
        elif aug == Augumentations.CYCLIC_SHIFT_3:
            wav_slice = cyclic_shift_augmentation(wav_slice, self.sr, frame_length=n_fft, hop_length=hop_length,
                                                  energy_threshold=0.0005,
                                                  shift=int(Augumentations.CYCLIC_SHIFT_3.split("_")[-1]))
        return wav_slice

    def get_spectrogram_shape(self):
        # Assuming 'path' in your dataframe points to the image files
        file_path = self.df['path'].iloc[0]

        # Load the image
        image = Image.open(file_path)

        # Optionally, resize the image if your model requires a specific input size.
        # For example, to resize to 224x224, you might do:
        resized_image = image.resize((224, 224))

        # Convert the image to a numpy array to work with it in numpy
        image_array = np.array(resized_image)

        # If the image is grayscale, it won't have a third dimension for channels.
        # We ensure it's in the format (height, width, channels) by adding missing dimensions.
        if len(image_array.shape) == 2:
            # Add a channels dimension and repeat the grayscale image across 3 channels if needed.
            image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
            # If the image already has a single channel, repeat it across 3 channels.
            image_array = np.repeat(image_array, 3, axis=2)

        # Return the shape of the image array
        return image_array.shape

    def _load_spectrogram(self, file_path):
        # Load the original spectrogram image
        image = Image.open(file_path)
        image_array = np.array(image)

        # If the image is grayscale, convert it to RGB by repeating the channels
        if len(image_array.shape) == 2:
            image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)

        # Assuming resize_spectrogram method exists and resizes the image as needed
        spectrogram = self.resize_spectrogram(image_array)

        # Normalize the spectrogram data to the range [0, 1]
        # Note: MinMaxScaler expects 2D data, so we need to reshape for scaler and then reshape back
        original_shape = spectrogram.shape
        spectrogram_flat = spectrogram.reshape(-1, 1)
        min_max = MinMaxScaler((0, 1))
        spectrogram_scaled = min_max.fit_transform(spectrogram_flat)
        spectrogram = spectrogram_scaled.reshape(original_shape)

        # Extract the original label from the file name and map it
        original_label = int(os.path.basename(file_path).split('_')[0])
        mapped_label = map_labels(original_label, self.label_mapping)  # Assuming map_labels method exists

        # Convert the mapped label into a one-hot encoded vector
        label_one_hot = tf.keras.utils.to_categorical(mapped_label, num_classes=self.num_classes)

        return spectrogram, label_one_hot

    def resize_spectrogram(self,spectrogram, target_shape=(224, 224, 3), method='linear'):
        """
        Resize a spectrogram to a target shape using specified interpolation method.

        :param spectrogram: Input spectrogram array of shape (height, width, channels).
        :param target_shape: Target shape tuple (new_height, new_width, new_channels).
        :param method: Interpolation method, 'linear' or 'spline'.
        :return: Resized spectrogram of shape target_shape.
        """
        resized_spectrogram = np.zeros(target_shape)

        # Iterate through each channel
        for channel in range(spectrogram.shape[-1]):
            channel_data = spectrogram[:, :, channel]

            if method == 'linear':
                # Calculate zoom factors
                zoom_factors = [target_shape[0] / channel_data.shape[0], target_shape[1] / channel_data.shape[1]]
                resized_channel = zoom(channel_data, zoom_factors, order=1)  # Linear interpolation

            elif method == 'spline':
                # Create coordinate grid for original and resized image
                old_x = np.arange(channel_data.shape[1])
                old_y = np.arange(channel_data.shape[0])
                new_x = np.linspace(0, channel_data.shape[1], target_shape[1])
                new_y = np.linspace(0, channel_data.shape[0], target_shape[0])

                spline = RectBivariateSpline(old_y, old_x, channel_data)
                resized_channel = spline(new_y, new_x)

            # Assign the resized channel back to the resized_spectrogram
            resized_spectrogram[:, :, channel] = resized_channel

        # If the original spectrogram is not in 3 channels, repeat the resized data across channels
        if spectrogram.shape[-1] != 3:
            resized_spectrogram = np.repeat(resized_spectrogram[:, :, np.newaxis], 3, axis=2)

        return resized_spectrogram

    def simple_resize_spectrogram(self,spectrogram, target_shape=(224, 224, 3)):
        return resize(spectrogram, target_shape, anti_aliasing=True, mode='reflect')

    def load_spectrogram_test(self, file_path):
        # Load the original spectrogram
        spectrogram = np.load(file_path)
        min_max = MinMaxScaler((0, 255))
        min_max.fit(spectrogram)
        spectrogram = min_max.transform(spectrogram)
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        # Next, repeat the data across this new dimension to get a shape of (128, 130, 3)
        spectrogram = np.repeat(spectrogram, 3, axis=-1)
        # Extract the original label from the file name and map it
        original_label = int(os.path.basename(file_path).split('_')[0])
        mapped_label = map_labels(original_label, self.label_mapping)

        # Convert the mapped label into a one-hot encoded vector
        label_one_hot = tf.keras.utils.to_categorical(mapped_label, num_classes=self.num_classes)

        return spectrogram, label_one_hot, self.resize_spectrogram(spectrogram)

    def k_fold_cross_validation(self, k, batch_size, epochs, callbacks, model_func, opt, model_name):
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []
        X = self.df['path']
        y = self.df['label']
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"Processing fold {fold + 1}/{k}")
            train_df = self.df.iloc[train_idx]
            val_df = self.df.iloc[val_idx]

            # Clear and process the data for the current fold
            self._clear_and_process_data(train_df, self.save_dir_train, model_name=model_name)
            self._clear_and_process_data(val_df, self.save_dir_val, is_test=True, model_name=model_name)
            time.sleep(70)
            # Prepare datasets
            self.train_files = [os.path.join(self.save_dir_train, f) for f in os.listdir(self.save_dir_train)]
            self.val_files = [os.path.join(self.save_dir_val, f) for f in os.listdir(self.save_dir_val)]

            self.unique_labels = extract_labels_from_filenames(self.save_dir_train)
            self.label_mapping = {original_label: new_label for new_label, original_label in
                                  enumerate(self.unique_labels)}
            train_dataset = self.create_dataset(self.train_files, batch_size)
            val_dataset = self.create_dataset(self.val_files, batch_size)
            model = model_func(model_name, opt)

            model.fit(train_dataset,
                      epochs=epochs,
                      verbose=1,
                      validation_data=val_dataset,
                      batch_size=batch_size,
                      callbacks=callbacks)
            test_loss, test_accuracy = model.evaluate(val_dataset)
            # Generate predictions for the validation dataset
            y_true = []
            y_pred = []

            for features, labels in val_dataset:
                # Generate predictions
                predictions = model.predict(features)

                # Convert one-hot encoded labels to class labels
                labels = np.argmax(labels, axis=1)

                # Convert predictions to class labels
                predicted_labels = np.argmax(predictions, axis=1)

                # Store true and predicted labels
                y_true.append(labels)
                y_pred.append(predicted_labels)

            # Concatenate results from all batches
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)

            label_one_hot_class_test = {}

            for label, values in self.label_mapping.items():
                label_one_hot_class_test[values] = self.label_one_hot_class[label]

            y_true_labels = [label_one_hot_class_test[idx] for idx in y_true]
            y_pred_labels = [label_one_hot_class_test[idx] for idx in y_pred]

            cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(label_one_hot_class_test.values()))

            # Optionally, plot the confusion matrix using seaborn
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_one_hot_class_test.values(),
                        yticklabels=label_one_hot_class_test.values())
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix for Fold {fold + 1}')
            plt.show()

            fold_results.append((test_accuracy, test_loss, model, cm))
            print(f"Finished processing fold {fold + 1}/{k}")

        return fold_results

    def _clear_and_process_data(self, df, save_dir, is_test=False, model_name=Models.RESNET):
        self._clear_directory(save_dir)
        self._process_and_save(df, save_dir, is_test=is_test, model_name=model_name)
        print("Processed: " + save_dir)

    def _clear_directory(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
