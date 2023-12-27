import numpy as np
import chess
import random

file_path = '/Users/osbor/OneDrive/Desktop/KDDM2/train_new.npy'
data = np.load(file_path)

# Extract frames after every move, which occurs every 260 frames
move_frames = data[:, :, ::260]

def create_board_from_data(data_frame):
    board = chess.Board()
    # Add logic to set pieces on the board based on your frame data
    # This might involve translating numeric values to piece types and positions
    return board

def predict_next_move(board):
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves) if legal_moves else None

# A simple text-based representation using python-chess
def display_board(board):
    print(board)

# Loop over the move frames and simulate the game
for i in range(move_frames.shape[2]):
    frame_data = move_frames[:, :, i]
    board = create_board_from_data(frame_data)
    
    # Display the current board
    display_board(board)
    
    # Predict the next move
    move = predict_next_move(board)
    if move:
        print(f"Predicted next move: {move.uci()}")
        board.push(move)
    else:
        print("No legal moves available or game over.")
    
    # Pause or wait for user input here if desired
    input("Press Enter to continue to the next move...")

    
    
