import numpy as np
import pandas as pd
import torch

from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from collections import deque
from statistics import mode


def mode_moving_window(labels, window_size, non_gesture_index):
    # Pad the input sequence to maintain the same length in the output
    padded_labels = [non_gesture_index] * (window_size // 2) + list(labels) + [non_gesture_index] * (window_size // 2)
    output = []
    window = deque(maxlen=window_size)
    
    # Initialize the window with the first window_size elements
    for i in range(window_size):
        window.append(padded_labels[i])
    
    # Process the remaining elements
    for i in range(window_size, len(padded_labels)):
        # Calculate the mode of the current window
        current_mode = mode(window)
        
        # Append the mode to the output
        output.append(current_mode)
        
        # Update the window by removing the oldest element and adding the new one
        window.popleft()
        window.append(padded_labels[i])
    
    return output


def postprocess(predictions, non_gesture_index, min_gesture_frames, max_gesture_frames):
    modified_predictions = mode_moving_window(predictions, 16, non_gesture_index) #predictions.copy()
    
    # ## Enforce constraint 2: a gesture shouldn't last less than 'min_gesture_frames'    
    # cursor = 0
    # prev_gest = modified_predictions[0]
    # while cursor < len(modified_predictions):
    #     current_gest = modified_predictions[cursor]
    #     if current_gest == non_gesture_index:
    #         cursor += 1
    #         continue
    #     else:
    #         gest_start = cursor
    #         n_frames = 1
    #         non_gesture_count = 0
    #         labels = []
    #         while cursor < len(modified_predictions)-1:
    #             cursor += 1
    #             n_frames += 1
                
    #             if modified_predictions[cursor] == non_gesture_index:
    #                 non_gesture_count += 1
    #             else:
    #                 non_gesture_count = 0
    #                 gest_end = cursor + 1
    #                 labels.append(modified_predictions[cursor])
                    
    #             if non_gesture_count > max_gesture_frames:
    #                 break
            
    #         vals, counts = np.unique(labels, return_counts=True)
    #         if len(counts) == 0 or (gest_end - gest_start + 1) < min_gesture_frames:
    #             dom_label = non_gesture_index
    #             gest_end = gest_start + 1
    #         else:
    #             dom_label = vals[list(counts).index(max(counts))]
    #         modified_predictions[gest_start:gest_end+1] = dom_label
            
    return modified_predictions


def segment_labels_sequence(labels):
    seg = []
    current_label = None
    start, end = None, None
    for i in range(len(labels)):
        if start is None:
            start = i
        if current_label is not None:
            if current_label != labels[i]:
                seg.append((current_label, start, i))
                current_label = labels[i]
                start = i + 1
        else:
            current_label = labels[i]  
    seg.append((current_label, start, i))
    return seg

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def exponential_moving_average(signal, alpha):
    return pd.Series(signal).ewm(alpha=alpha, adjust=False).mean().values

def median_filter(signal, window_size):
    return medfilt(signal, window_size)

def gaussian_smooth(signal, sigma):
    return gaussian_filter1d(signal, sigma=sigma)

def pad_sequence(seq, target_length):
    b, t, n, d = seq.shape
    if t >= target_length:
        return seq[:, :target_length]
    pad_len = target_length - t
    padding = torch.zeros((b, pad_len, n, d), dtype=seq.dtype).to(seq.device)
    return torch.cat([seq, padding], dim=1)


from torchaudio.functional import edit_distance

def levenshtein_accuracy(pred_labels, true_labels):
    """
    Compute the Levenshtein accuracy between two arrays of labels.
    """
    total_distance = 0
    total_length = 0
    
    for pred, true in zip(pred_labels, true_labels):
        # pred_tensor = torch.tensor(pred, device='cuda')
        # true_tensor = torch.tensor(true, device='cuda')
        distance = edit_distance(pred, true)
        total_distance += distance
        total_length += max(len(pred), len(true))
    
    accuracy = 1 - total_distance / total_length
    return accuracy