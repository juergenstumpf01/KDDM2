import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load the 'repeat last state' baseline file
repeat_last_state = np.load("25_DeepBlue.npy")

# Update with final prediction file
# Load your prediction data to find the shape
predictions = np.load("prediction.npy")  

# Save this array to a .npy file
# np.save("zeros_prediction.npy", zeros_array)
zeros_prediction = np.load("dummy_test_dataset.npy")

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
print(" ")


# Split the data into training and testing sets with 80/20 split
train_data, test_data = train_test_split(predictions, test_size=0.2, random_state=42)

train_data_reshaped = train_data.reshape(train_data.shape[0], -1)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)


# Calculate the cosine similarity
cos_sim = cosine_similarity(train_data_reshaped, test_data_reshaped)

print("Cosine Similarity:")
print(cos_sim)