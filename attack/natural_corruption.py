import numpy as np

def generate_random(x:np.ndarray, mean, sigma):
    noise_mean = mean  # Mean of the noise
    noise_stddev = sigma  # Standard deviation of the noise
    num_rows, num_cols = x.shape  # Get the shape of your array
    columns = [7,62,65,69,67,48,73,43,42,21,20,8,10,11,49,50,51,52,53,54,55,56,75,76,77,78,79,80,81,82]
    for i in range(22,36):
        columns.append(i)
    # Create random noise with the same shape as the selected columns
    noise = np.random.normal(noise_mean, noise_stddev, size=(num_rows, len(columns)))  
    x[:, columns] += noise
    return x