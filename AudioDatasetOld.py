import os
import random
import shutil
import time
import cv2
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift, ifft2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from keras.applications.inception_v3 import preprocess_input
import Augumentations
import Models
from audio_globals import n_mels, slice_lenght, overlap, n_mfcc, n_fft, hop_length
from librosa.feature import melspectrogram
import matplotlib.colors as colors
import seaborn as sns


def set_last_n_layers_trainable(model, n):

    for layer in model.layers:
        layer.trainable = False

    # Set the last n layers as trainable
    for layer in model.layers[-n:]:
        layer.trainable = True

def combine_histories(history_main, history_fine):
    combined_history = {key: [] for key in list(history_main.history.keys()) + list(history_fine.history.keys())}

    # Fill combined history with main training history
    for key, values in history_main.history.items():
        combined_history[key].extend(values)

    # Append fine-tuning history
    for key, values in history_fine.history.items():
        if key in combined_history:
            combined_history[key].extend(values)
        else:
            combined_history[key] = values

    return combined_history
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


def plot_history(history, fold):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold + 1} - Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


class AudioDataset:

    def __init__(self, csv_file, save_dir, sr=44100, feature_type='melspectogram', augumentations=None,interpolate=False, interpolation=cv2.INTER_LINEAR,model_name=""):
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
        self.interpolate=interpolate
        self.interpolation = interpolation
        self.model_name = model_name

    def _load_csv(self):
        # Load the entire CSV file
        full_df = pd.read_csv(self.csv_file)

        # Identify 8 unique classes - this assumes 'label' contains numerical class identifiers
        unique_classes = ["chainsaw","clock_tick","crackling_fire","crying_baby","dog","helicopter","rooster","rain","sneezing","sea_waves"]

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

    def _process_and_save(self, df, save_dir, is_test=False):
        for _, row in df.iterrows():
            file_path = row['path']
            label = row['label']
            wav, _ = librosa.load(file_path, sr=self.sr)
            wav = peak_normalize(wav)
            slice_lenght_samples = int(slice_lenght*self.sr)
            overlap_amount = slice_lenght_samples * (overlap / 100)
            step_size = slice_lenght_samples - overlap_amount
            num_slices = int(max(1, 1 + (len(wav) - slice_lenght_samples) // step_size))

            augmentations = self.augumentations if not is_test else [
                'original']
            for slice_num in range(num_slices):
                start = int(slice_num * step_size)
                end = int(start + slice_lenght_samples)
                wav_slice = wav[start:end]

                if len(wav_slice) < slice_lenght_samples:
                    wav_slice = np.pad(wav_slice, (0, slice_lenght_samples - len(wav_slice)), mode='constant')

                for aug in augmentations:
                    augmented_slice = self.apply_augmentation(wav_slice, aug, df,label)
                    augmented_slice = augmented_slice[:slice_lenght_samples]
                    if len(augmented_slice) < slice_lenght_samples:
                        shortfall = slice_lenght_samples - len(augmented_slice)
                        augmented_slice = np.pad(augmented_slice, (0, shortfall), mode='constant')

                    feature = self._extract_feature(augmented_slice, self.sr)
                    feature = librosa.util.normalize(feature)
                    file_name = f'{label}_{os.path.basename(file_path).split(".")[0]}_{slice_num}_{aug}.npy'
                    np.save(os.path.join(save_dir, file_name), feature)

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
            feature = librosa.power_to_db(S)
        elif self.feature_type == 'spectrogram':
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024)) ** 2
            feature = librosa.power_to_db(S)
        elif self.feature_type == 'mfcc':
            feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        else:
            raise ValueError("Unsupported feature type")
        if self.model_name == Models.DENSNET:
            min_shape = (75, 75)
            pad_width = [(0, max(0, min_shape[0] - feature.shape[0])), (0, max(0, min_shape[1] - feature.shape[1]))]
            if pad_width[0][1] > 0 or pad_width[1][1] > 0:
                pad_value = np.min(feature)
                feature = np.pad(feature, pad_width, 'constant', constant_values=pad_value)

        return feature

    def apply_augmentation(self, wav_slice, aug, train_df, label):
        if aug == 'pitch_shift':
            pitch_shift = np.random.randint(-10, 10)
            wav_slice = librosa.effects.pitch_shift(wav_slice, sr=self.sr, n_steps=pitch_shift)
        elif aug == 'time_stretch':
            stretch_rate = np.random.uniform(0.6, 1.9)
            wav_slice = librosa.effects.time_stretch(wav_slice, rate=stretch_rate)
        elif aug == 'noise_addition':
            same_label_df = train_df[train_df['label'] == label]
            if not same_label_df.empty:
                random_file = same_label_df.sample(1).iloc[0]
                file_path = random_file['path']
                random_audio, _ = librosa.load(file_path, sr=self.sr)
                wav_slice = wav_slice + random_audio[0:len(wav_slice)]
        return wav_slice

    def get_spectrogram_shape(self):
        file_path = self.df['path'].iloc[0]
        wav, _ = librosa.load(file_path, sr=self.sr)

        wav_slice = wav[0:int(slice_lenght * self.sr)]
        feature = self._extract_feature(wav_slice, self.sr)
        feature = librosa.util.normalize(feature)
        feature = np.expand_dims(feature, axis=-1)

        feature = np.repeat(feature, 3, axis=-1)
        if self.interpolate:
            feature = self.interpolate_image(feature)
        # feature = self.resize_spectrogram(feature)
        return feature.shape

    def resize_image_with_fourier(self,image, new_size):
        image_2d = image[:, :, 0]
        fft_image = fft2(image_2d)
        fft_shifted = fftshift(fft_image)
        size_diff = np.array(new_size) - np.array(image_2d.shape)

        if np.all(size_diff >= 0):
            pad_before = size_diff // 2
            pad_after = size_diff - pad_before
            fft_resized = np.pad(fft_shifted, [(pad_before[0], pad_after[0]), (pad_before[1], pad_after[1])],
                                 mode='constant')
        else:
            crop_before = np.abs(size_diff) // 2
            crop_after = np.abs(size_diff) - crop_before
            fft_resized = fft_shifted[crop_before[0]:-crop_after[0], crop_before[1]:-crop_after[1]]
        fft_resized_shifted_back = ifftshift(fft_resized)
        resized_image = ifft2(fft_resized_shifted_back)
        resized_image_real = np.real(resized_image)
        resized_image_real = np.expand_dims(resized_image_real, axis=-1)
        resized_image_real = np.repeat(resized_image_real, 3, axis=-1)
        return resized_image_real

    def zero_padding_spectrogram(self,image, new_size=(224, 224)):
        image_2d = image[:, :, 0]

        pad_height = (new_size[0] - image_2d.shape[0], 0)
        pad_width = (new_size[1] - image_2d.shape[1], 0)

        pad_height = (pad_height[0] // 2, pad_height[0] - pad_height[0] // 2)
        pad_width = (pad_width[0] // 2, pad_width[0] - pad_width[0] // 2)
        min_value = np.min(image_2d)

        # Pad the spectrogram
        padded_spectrogram = np.pad(image_2d, (pad_height, pad_width), 'constant', constant_values=min_value)
        padded_spectrogram = np.expand_dims(padded_spectrogram, axis=-1)
        padded_spectrogram = np.repeat(padded_spectrogram, 3, axis=-1)
        return padded_spectrogram
    def interpolate_image(self,image, output_shape=(224, 224)):
        if self.interpolation == "ZERO":
            return self.zero_padding_spectrogram(image,output_shape)
        if self.interpolation == "GONIOMETRIC":
            return self.resize_image_with_fourier(image,output_shape)
        if self.model_name == Models.DENSNET:
            output_shape = (299, 299)
        return cv2.resize(image, output_shape, interpolation=self.interpolation)

    def _load_spectrogram(self, file_path):
        spectrogram = np.load(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        spectrogram = np.repeat(spectrogram, 3, axis=-1)

        original_label = int(os.path.basename(file_path).split('_')[0])
        mapped_label = map_labels(original_label, self.label_mapping)
        if self.interpolate:
            spectrogram = self.interpolate_image(spectrogram)
        label_one_hot = tf.keras.utils.to_categorical(mapped_label, num_classes=self.num_classes)
        return spectrogram, label_one_hot


    def k_fold_cross_validation(self, k, batch_size, epochs, callbacks, model_func,model_name, val_split=0.15,fine_tune_at=None, fine_tune_epochs=0):
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        self.model_name=model_name
        fold_results = []
        X = self.df['path']
        y = self.df['label']
        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            print(f"Processing fold {fold + 1}/{k}")
            train_df = self.df.iloc[train_idx]
            test_df = self.df.iloc[test_idx]

            # Splitting the training data to create a new validation set
            train_df, val_df = train_test_split(train_df, test_size=val_split, stratify=train_df['label'],
                                                random_state=42)

            # Clear and process the data for the current fold
            self._clear_and_process_data(train_df, self.save_dir_train)
            self._clear_and_process_data(val_df, self.save_dir_val, is_test=True)  # Now is_test is False for val data
            self._clear_and_process_data(test_df, self.save_dir_test, is_test=True)
            time.sleep(150)  # Assuming this is necessary for some file system operations to complete

            # Prepare datasets
            train_files = [os.path.join(self.save_dir_train, f) for f in os.listdir(self.save_dir_train)]
            val_files = [os.path.join(self.save_dir_val, f) for f in os.listdir(self.save_dir_val)]
            test_files = [os.path.join(self.save_dir_test, f) for f in os.listdir(self.save_dir_test)]

            train_dataset = self.create_dataset(train_files, batch_size)
            val_dataset = self.create_dataset(val_files, batch_size)
            test_dataset = self.create_dataset(test_files, batch_size)

            opt = Adam(learning_rate=0.001)
            model = model_func(model_name,opt)
            start_time = time.time()

            history=model.fit(train_dataset,
                      epochs=epochs,
                      verbose=1,
                      validation_data=val_dataset,
                      batch_size=batch_size,
                      callbacks=callbacks)

            if fine_tune_at is not None:
                n_layers_to_unfreeze=0
                if model_name == Models.VGG16:
                    n_layers_to_unfreeze = 6  # The last convolutional block
                elif model_name == Models.RESNET:
                    n_layers_to_unfreeze = 15  # The last residual block
                elif model_name == Models.DENSNET:
                    n_layers_to_unfreeze = 9  # Approximation of the last 2-3 Inception modules
                set_last_n_layers_trainable(model,n_layers_to_unfreeze)

                adam_fine = Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                 amsgrad=False)
                model.compile(optimizer=adam_fine,
                              # Assuming `opt` has a learning rate attribute
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

                history_fine=model.fit(train_dataset,
                          epochs=40,
                          verbose=1,
                          validation_data=val_dataset,
                          batch_size=batch_size,
                          callbacks=callbacks
                                       )
                plot_history(combine_histories(history,history_fine),fold)
            training_time = time.time() - start_time

            # Evaluate the model on the test dataset
            test_loss, test_accuracy = model.evaluate(test_dataset)
            # Generate predictions for the validation dataset
            y_true = []
            y_pred = []

            for features, labels in test_dataset:
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

            fold_results.append((test_accuracy, test_loss, combine_histories(history,history_fine), cm,training_time))
            print(f"Finished processing fold {fold + 1}/{k}")

        return fold_results

    def _clear_and_process_data(self, df, save_dir, is_test=False):
        self._clear_directory(save_dir)
        self._process_and_save(df, save_dir, is_test=is_test)
        print("Processed: " + save_dir)

    def _clear_directory(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
