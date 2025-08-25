#!/usr/bin/env python3
"""
Script to load, preprocess, and prepare pump sensor data for CNN training.
Handles data loading, cleaning, sliding window creation, normalization, and balanced splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data(data_path="data/"):
    """
    Load the pump sensor dataset from the local data folder.
    
    Args:
        data_path (str): Path to the data folder
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Look for CSV files in the data directory
    data_files = list(Path(data_path).glob("*.csv"))
    
    if not data_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    
    # Load the first CSV file found
    data_file = data_files[0]
    print(f"Loading data from: {data_file}")
    
    # Load the data
    df = pd.read_csv(data_file)
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def clean_data(df):
    """
    Clean the data and map machine_status to numerical values.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Cleaning data...")
    
    # Check for missing values
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    # Remove rows with missing values
    df_clean = df.dropna()
    print(f"Shape after removing missing values: {df_clean.shape}")
    
    # Map machine_status to numerical values
    status_mapping = {'NORMAL': 0, 'WARNING': 1, 'BROKEN': 2}
    df_clean['machine_status'] = df_clean['machine_status'].map(status_mapping)
    
    # Check the distribution of classes
    print(f"Class distribution:")
    print(df_clean['machine_status'].value_counts().sort_index())
    
    return df_clean

def create_sliding_windows(df, window_size=256, step_size=1):
    """
    Create sliding windows over the sensor data.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        window_size (int): Size of each window
        step_size (int): Step size between windows
        
    Returns:
        tuple: (windows, labels, timestamps)
    """
    print(f"Creating sliding windows of size {window_size}...")
    
    # Get sensor columns (exclude non-sensor columns)
    sensor_columns = [col for col in df.columns if col not in ['timestamp', 'machine_status']]
    
    windows = []
    labels = []
    timestamps = []
    
    # Create sliding windows
    for i in range(0, len(df) - window_size + 1, step_size):
        window_data = df.iloc[i:i + window_size][sensor_columns].values
        window_label = df.iloc[i + window_size - 1]['machine_status']  # Label from last sample in window
        window_timestamp = df.iloc[i + window_size - 1]['timestamp']
        
        windows.append(window_data)
        labels.append(window_label)
        timestamps.append(window_timestamp)
    
    windows = np.array(windows)
    labels = np.array(labels)
    
    print(f"Created {len(windows)} windows with shape: {windows.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return windows, labels, timestamps

def normalize_windows(windows):
    """
    Normalize each sensor channel using z-score normalization.
    
    Args:
        windows (np.ndarray): Array of windows with shape (n_windows, window_size, n_sensors)
        
    Returns:
        np.ndarray: Normalized windows
    """
    print("Normalizing sensor channels...")
    
    n_windows, window_size, n_sensors = windows.shape
    
    # Reshape to (n_windows * window_size, n_sensors) for easier normalization
    windows_reshaped = windows.reshape(-1, n_sensors)
    
    # Apply z-score normalization to each sensor channel
    scaler = StandardScaler()
    windows_normalized = scaler.fit_transform(windows_reshaped)
    
    # Reshape back to original shape
    windows_normalized = windows_normalized.reshape(n_windows, window_size, n_sensors)
    
    print(f"Normalization completed. Mean: {np.mean(windows_normalized):.6f}, Std: {np.std(windows_normalized):.6f}")
    
    return windows_normalized

def split_data_chronologically(windows, labels, timestamps, train_ratio=0.7, val_ratio=0.15):
    """
    Split the data chronologically into train, validation, and test sets.
    
    Args:
        windows (np.ndarray): Normalized windows
        labels (np.ndarray): Labels
        timestamps (list): Timestamps
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        
    Returns:
        tuple: (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    print("Splitting data chronologically...")
    
    n_samples = len(windows)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Split data chronologically
    train_windows = windows[:train_end]
    val_windows = windows[train_end:val_end]
    test_windows = windows[val_end:]
    
    train_labels = labels[:train_end]
    val_labels = labels[train_end:val_end]
    test_labels = labels[val_end:]
    
    print(f"Train set: {len(train_windows)} samples")
    print(f"Validation set: {len(val_windows)} samples")
    print(f"Test set: {len(test_windows)} samples")
    
    return train_windows, val_windows, test_windows, train_labels, val_labels, test_labels

def balance_classes(windows, labels, target_class=0):
    """
    Balance classes by undersampling the majority class (NORMAL).
    
    Args:
        windows (np.ndarray): Input windows
        labels (np.ndarray): Input labels
        target_class (int): Class to undersample (0 for NORMAL)
        
    Returns:
        tuple: (balanced_windows, balanced_labels)
    """
    print("Balancing classes by undersampling...")
    
    # Get indices for each class
    class_indices = {}
    for class_label in np.unique(labels):
        class_indices[class_label] = np.where(labels == class_label)[0]
    
    # Find the minority class size (excluding the target class)
    minority_size = min([len(indices) for label, indices in class_indices.items() if label != target_class])
    
    # Undersample the target class to match minority class size
    if len(class_indices[target_class]) > minority_size:
        undersampled_indices = np.random.choice(
            class_indices[target_class], 
            size=minority_size, 
            replace=False
        )
        class_indices[target_class] = undersampled_indices
    
    # Combine all balanced indices
    balanced_indices = np.concatenate(list(class_indices.values()))
    
    # Shuffle the indices
    np.random.shuffle(balanced_indices)
    
    balanced_windows = windows[balanced_indices]
    balanced_labels = labels[balanced_indices]
    
    print(f"Balanced dataset shape: {balanced_windows.shape}")
    print(f"Balanced label distribution: {np.bincount(balanced_labels)}")
    
    return balanced_windows, balanced_labels

def main():
    """
    Main function to execute the data preprocessing pipeline.
    """
    print("Starting pump sensor data preprocessing pipeline...")
    
    try:
        # 1. Load data
        df = load_data()
        
        # 2. Clean data and map machine_status
        df_clean = clean_data(df)
        
        # 3. Create sliding windows
        windows, labels, timestamps = create_sliding_windows(df_clean, window_size=256)
        
        # 4. Normalize sensor channels
        windows_normalized = normalize_windows(windows)
        
        # 5. Split data chronologically
        train_windows, val_windows, test_windows, train_labels, val_labels, test_labels = split_data_chronologically(
            windows_normalized, labels, timestamps
        )
        
        # 6. Balance classes by undersampling NORMAL
        train_windows_balanced, train_labels_balanced = balance_classes(train_windows, train_labels, target_class=0)
        
        # Save preprocessed data
        output_dir = Path("preprocessed_data")
        output_dir.mkdir(exist_ok=True)
        
        np.save(output_dir / "train_windows.npy", train_windows_balanced)
        np.save(output_dir / "train_labels.npy", train_labels_balanced)
        np.save(output_dir / "val_windows.npy", val_windows)
        np.save(output_dir / "val_labels.npy", val_labels)
        np.save(output_dir / "test_windows.npy", test_windows)
        np.save(output_dir / "test_labels.npy", test_labels)
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Data saved to: {output_dir}")
        print(f"Final dataset shapes:")
        print(f"  Train: {train_windows_balanced.shape}")
        print(f"  Validation: {val_windows.shape}")
        print(f"  Test: {test_windows.shape}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()