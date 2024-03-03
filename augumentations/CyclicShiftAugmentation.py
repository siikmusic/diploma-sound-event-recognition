import numpy as np
import librosa
import IPython.display as ipd
import pandas as pd
import noisereduce as nr


def cyclic_shift(audio, shift):
    """Cyclically shift the audio signal by `shift` samples."""
    return np.roll(audio, shift)


def reconstruct_signal_from_windowed_frames(windowed_frames, frame_length, hop_length, original_length):
    """Reconstruct an audio signal from windowed frames."""
    # Initialize the reconstructed signal array with zeros
    reconstructed_signal = np.zeros(original_length)

    # Number of frames
    num_frames = windowed_frames.shape[1]

    for i in range(num_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        # Add the windowed frame back, taking care of boundary conditions
        reconstructed_signal[start_idx:end_idx] += windowed_frames[:, i]

    return reconstructed_signal


def apply_window(audio, event_position, frame_length, hop_length):
    """Apply windowing based on the event position and energy distribution."""
    # Example: Simple windowing, adjust according to your energy analysis
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    windowed_frames = np.hanning(frame_length).reshape(-1, 1) * frames
    return reconstruct_signal_from_windowed_frames(windowed_frames, frame_length, hop_length, len(audio))


def augment_audio(audio, start_time, end_time, sr=44100, shift=1):
    """Load audio, perform cyclic shift based on event duration, and apply windowing."""

    # Convert start and end times to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Calculate event duration in samples
    event_duration_samples = end_sample - start_sample

    # Example shift: Move the event to the middle of the audio file
    # Calculate new start position as halfway through the audio
    new_start_sample = end_sample + event_duration_samples * shift
    # Calculate shift amount
    shift_amount = new_start_sample - start_sample

    # Perform cyclic shift
    shifted_audio = cyclic_shift(audio, shift_amount)
    windowed_audio = apply_window(shifted_audio, new_start_sample, frame_length=1024, hop_length=512)

    return windowed_audio


def detect_event_positions(y, frame_length=1024, hop_length=512, energy_threshold=0.02, sr=44100):
    # Noise reduce
    y = nr.reduce_noise(y=y, sr=sr)
    # Calculate short-term energy
    energy = np.array([np.sum(np.abs(y[i:i + frame_length]) ** 2) for i in range(0, len(y), hop_length)])

    # Normalize energy
    energy = energy / np.max(energy)

    # Detect frames exceeding the threshold
    event_frames = np.where(energy > energy_threshold)[0]

    # Convert event frames to time
    event_times = librosa.frames_to_time(event_frames, sr=sr, hop_length=hop_length)

    return event_times


def cyclic_shift_augmentation(audio, sr=44100, frame_length=1024, hop_length=512, energy_threshold=0.0005, shift=1):
    position = detect_event_positions(audio, frame_length=frame_length, hop_length=hop_length,
                                      energy_threshold=energy_threshold, sr=sr)
    augmented_audio = augment_audio(audio, start_time=position[0], end_time=position[-1], sr=sr, shift=shift)
    return augmented_audio
