import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load the hand key point data from a CSV file
data = np.loadtxt('hand_keypoints.csv', delimiter=',')

# Define the window size for feature analysis
window_size = 10

# Define the features to compute
features = ['mean_x', 'std_x', 'mean_y', 'std_y', 'mean_vel', 'std_vel', 
            'mean_acc', 'std_acc', 'mean_angle', 'std_angle', 'mean_dist', 
            'std_dist', 'mean_freq', 'std_freq']

# Define the labels for each window based on tremor severity
labels = np.array(['severe', 'moderate', 'mild'])

# Initialize the feature matrix and label vector
X = np.zeros((data.shape[0] - window_size, len(features)))
y = np.zeros(data.shape[0] - window_size)

# Compute the features for each window of frames
for i in range(data.shape[0] - window_size):
    window_data = data[i:i+window_size]
    
    # Compute x and y means and standard deviations
    X[i,0] = np.mean(window_data[:,0])
    X[i,1] = np.std(window_data[:,0])
    X[i,2] = np.mean(window_data[:,1])
    X[i,3] = np.std(window_data[:,1])
    
    # Compute velocities and accelerations using central differences
    dx = np.diff(window_data[:,0])
    dy = np.diff(window_data[:,1])
    dt = np.ones_like(dx)
    dt[0] = 2
    vel = np.sqrt(dx**2 + dy**2) / dt
    acc = np.diff(vel)
    X[i,4] = np.mean(vel)
    X[i,5] = np.std(vel)
    X[i,6] = np.mean(acc)
    X[i,7] = np.std(acc)
    
    # Compute angles and distances between adjacent points
    diff = window_data[1:] - window_data[:-1]
    angles = np.arctan2(diff[:,1], diff[:,0])
    angles[angles < 0] += 2*np.pi
    dists = np.sqrt(np.sum(diff**2, axis=1))
    X[i,8] = np.mean(angles)
    X[i,9] = np.std(angles)
    X[i,10] = np.mean(dists)
    X[i,11] = np.std(dists)
    
    # Compute frequency using Fourier transform
    freq = np.fft.fft(window_data[:,0] + 1j*window_data[:,1])
    freq = np.abs(freq[:window_size//2]) / window_size
    peaks, _ = find_peaks(freq)
    if len(peaks) > 0:
        X[i,12] = np.mean(peaks)
        X[i,13] = np.std(peaks)
    
    # Assign the label based on the severity of the tremor in the window
    if np.mean(vel) > 100 and np.mean(acc) > 500:
        y[i] = 0  # severe
    elif np.mean(vel) > 50 and np.mean(acc) > 200:
        y[i] = 1  # moderate
    else:
        y[i] = 2  #

