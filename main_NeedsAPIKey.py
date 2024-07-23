import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import SpotifyDataPull as sdp
import LLayerNeuralNet as llnn


### CONSTANTS ###
LEARNING_RATE = 0.025
NUM_ITERATIONS = 3000
PRINT_COST = True

# Spotify API credentials
CLIENT_ID = #your client id
CLIENT_SECRET = #your client secret
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = 'user-library-read'

# File paths
CSV_DATASET = 'Spotify Songs Archive/spotify_songs.csv'  # Path to your CSV file
LIKED_SONGS_FILE = 'liked_songs.json'  # File to store liked songs

# Main execution
print("Loading CSV dataset...")
csv_data = sdp.load_csv_dataset(CSV_DATASET)

print("Checking for saved liked songs...")
liked_songs = sdp.load_liked_songs(LIKED_SONGS_FILE)

if liked_songs is None:
    print("Fetching liked songs from Spotify API...")
    liked_songs = sdp.get_liked_songs_from_api(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE)
    sdp.save_liked_songs(liked_songs, LIKED_SONGS_FILE)

print("Preparing final dataset...")
df = sdp.prepare_dataset(csv_data, liked_songs)

# Prepare features for neural network
features = ['duration_ms', 'track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[features].values
y = df['liked'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(y.shape[0],1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

#Model layers
layers_dims = [X_train.shape[0], 7, 5, 2, 1] #  4-layer model

#Create the model and iterate
parameters, costs = llnn.L_layer_model(X_train, y_train, layers_dims, LEARNING_RATE, NUM_ITERATIONS, PRINT_COST)
llnn.plot_costs(costs, LEARNING_RATE)

#Use model to predict training set
pred_train = llnn.predict(X_train, y_train, parameters)

#Use model to predict test set
pred_test = llnn.predict(X_test, y_test, parameters)