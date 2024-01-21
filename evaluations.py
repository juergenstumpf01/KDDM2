import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from ipywidgets import interact
import pandas as pd
import seaborn as sns
from scipy.signal import lfilter
from scipy.stats import linregress
import chess
import chess.svg
from IPython.display import display, SVG, clear_output
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean


split_index = 7020

data = np.load("train_new.npy")
train_data = data[:,:,0:split_index]
test_data = data[:,:,split_index:12220]

def plot_time_series(spatial_index, data):
    time_series = data[spatial_index[0], spatial_index[1], :]
    plt.figure(figsize=(10, 5))  # Set the figure size

    plt.plot(range(1, data.shape[2] + 1), time_series, label=f'Spatial Index {spatial_index}', linewidth=0.2)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Time Series for Spatial Index {spatial_index}')
    plt.legend()
    plt.ylim([-3, 3])
    plt.show()

data_new = data.copy()
block = 0
res = np.zeros((8, 8, 47))
for row in range(8):
    
    for col in range(8):

        block = np.clip(data_new[row,col,:], -1, 1)
        for n in range(47):
            res[row,col,n] = np.rint(np.mean(block[260*n:260*n+259]))


result = res


def get_linear_funciton(data, spatial_index, plot=False):

    selected_data = data[spatial_index[0], spatial_index[1]]

    y_values = selected_data 
    x_values = np.arange(len(y_values))

    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    #print(slope)

    linear_function = lambda x: slope * x + intercept

    x_linear_function = np.linspace(0, len(y_values) - 1, num=1000)
    y_linear_function = linear_function(x_linear_function)
    

    if plot:
        plt.figure(figsize=(12, 8)) 
        plt.plot(range(1, len(selected_data) + 1), selected_data, label=f'Spatial Index {spatial_index}', linewidth=0.2)
        plt.plot(x_linear_function, y_linear_function, color='red', label=f'Linear Function: y = {slope:.6f}x + {intercept:.2f}')

        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'Time Series for Spatial Index {spatial_index}')
        plt.legend()
        plt.show()
    return slope, intercept

clean_moves = result.copy()
clean_data = train_data.copy()
cleaned_data = np.empty((8, 8), dtype=object)

# substract moves to get linear data
for i in range(27):
    for j in range(260):
        clean_data[:,:,260*i+j] -= clean_moves[:,:,i]

# remove outliers
mask = (clean_data >= 10) | (clean_data <= -10)
clean_data[mask] = np.nan

# remove nan

for i in range(clean_data.shape[0]):
    for j in range(clean_data.shape[1]):
        tmp = clean_data[i,j]
        nan_indices = np.isnan(tmp)
        tmp = tmp[~nan_indices] 
        cleaned_data[i,j] = tmp



get_linear_funciton(cleaned_data,(1,1), True)

linear_functions = np.zeros((8,8,2))

##_______________________________________

improvedpred = np.zeros((8,8,5200))
for i in range(8):
    for j in range(8):
        improvedpred[i,j] = np.repeat(result[i,j,27:],260)
print(improvedpred.shape)

for i in range(cleaned_data.shape[0]):
    for j in range(cleaned_data.shape[1]):
        linear_functions[i,j] = get_linear_funciton(cleaned_data, (i,j))



for i in range(improvedpred.shape[0]):
    for j in range(improvedpred.shape[1]):
        for x in range(improvedpred.shape[2]):
            improvedpred[i,j,x] = improvedpred[i,j,x] + linear_functions[i,j,0]*(x+7020) + linear_functions[i,j,1]

print(improvedpred.shape)
pred_moves_with_trend = improvedpred


pred_moves_without_trend = np.zeros((8,8,12220-split_index))
for i in range(8):
    for j in range(8):
        pred_moves_without_trend[i,j] =  np.repeat(result[i,j,27:],260)






pred_always_zero = np.zeros((8,8,12220-split_index))
pred_repeat_last_move = np.zeros((8,8,12220-split_index))
for i in range(12220-split_index):   
    pred_repeat_last_move[:,:,i] = data[:,:,split_index]



def get_cosine_sim(train, test):
    mean_cosim = 0
    for i in range(8):
        for j in range(8):
            dist = cosine_similarity(train[i,j].T, test[i,j].T)[0,0]
            mean_cosim += np.mean(dist)

    return mean_cosim / 64

#plot_time_series([1,1], pred_always_zero)
#plot_time_series([1,1], pred_repeat_last_move)
#plot_time_series([1,1], pred_moves_without_trend)
#plot_time_series([1,1], pred_moves_with_trend)



print(f"pred_always_zero_reshaped: {get_cosine_sim(pred_always_zero, test_data)}")

print(f"pred_pred_repeat_last_move: {get_cosine_sim(pred_repeat_last_move, test_data)}")

print(f"pred_pred_moves_without_trend: {get_cosine_sim(pred_moves_without_trend, test_data)}")

print(f"pred_pred_moves_with_trend: {get_cosine_sim(pred_moves_with_trend, test_data)}")



#def dhfalsjhd():

    # Initialize variables to accumulate theMSE and MAE
    #total_mse = 0

    # Calculate MSE and MAE for each 2D slice
    #num_slices = predictions.shape[2]
    #for i in range(num_slices):
        #total_mse += mean_squared_error(repeat_last_state[:,:,i], predictions[:,:,i])

    # Calculate the average MSE and MAE
    #average_mse = total_mse / num_slices

    #print("Repeat Last State Baseline:")
    #print(f"Average MSE: {average_mse}")

    # Initialize variables to accumulate theMSE and MAE
    #total_mse = 0

    # Calculate MSE and MAE for each 2D slice
    #num_slices = predictions.shape[2]
    #for i in range(num_slices):
    #    total_mse += mean_squared_error(zeros_prediction[:,:,i], predictions[:,:,i])

    # Calculate the average MSE and MAE
    #average_mse = total_mse / num_slices

    #print("Always Predict Zero Baseline:")
    #print(f"Average MSE: {average_mse}")
    #print(" ")


    # Split the data into training and testing sets with 80/20 split
    #train_data, test_data = train_test_split(predictions, test_size=0.2, random_state=42)

    #train_data_reshaped = train_data.reshape(train_data.shape[0], -1)
    #test_data_reshaped = test_data.reshape(test_data.shape[0], -1)


    # Calculate the cosine similarity
    #cos_sim = cosine_similarity(train_data_reshaped, test_data_reshaped)

    #print("Cosine Similarity:")
    #print(cos_sim)