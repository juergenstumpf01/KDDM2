import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the prediction file (replace 'prediction.npy' with your actual prediction file's name)
predictions = np.load("prediction.npy")

# Load the 'repeat last state' baseline file
repeat_last_state = np.load("25_DeepBlue.npy")

# Load your prediction data to find the shape
predictions = np.load("prediction.npy")  # replace with your actual file if different

# Get the shape of your predictions
shape_of_predictions = predictions.shape

# Create an array of zeros with the same shape
zeros_array = np.zeros(shape_of_predictions)

# Save this array to a .npy file
# np.save("zeros_prediction.npy", zeros_array)
zeros_prediction = np.load("zeros_prediction.npy")

# Initialize variables to accumulate theMSE and MAE
total_mse = 0

# Calculate MSE and MAE for each 2D slice
num_slices = predictions.shape[2]
for i in range(num_slices):
    total_mse += mean_squared_error(repeat_last_state[:,:,i], predictions[:,:,i])

# Calculate the average MSE and MAE
average_mse = total_mse / num_slices

print("Repeat Last State Baseline:")
print(f"Average MSE: {average_mse}")

# Initialize variables to accumulate theMSE and MAE
total_mse = 0

# Calculate MSE and MAE for each 2D slice
num_slices = predictions.shape[2]
for i in range(num_slices):
    total_mse += mean_squared_error(zeros_prediction[:,:,i], predictions[:,:,i])

# Calculate the average MSE and MAE
average_mse = total_mse / num_slices

print("Always Predict Zero Baseline:")
print(f"Average MSE: {average_mse}")
