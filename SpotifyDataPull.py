import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
import pandas as pd

def load_csv_dataset(file_path):
    return pd.read_csv(file_path)

def save_liked_songs(liked_songs, filename):
    with open(filename, 'w') as f:
        json.dump(liked_songs, f)
    print(f"Liked songs saved to {filename}")

def load_liked_songs(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found. Will fetch liked songs from Spotify API.")
        return None

def get_liked_songs_from_api(client_id, client_secret, redirect_uri, scope):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id, 
                                                   client_secret, 
                                                   redirect_uri, 
                                                   scope))
    liked_songs = []
    offset = 0
    
    while True:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        if len(results['items']) == 0:
            break
        
        for item in results['items']:
            track = item['track']
            liked_songs.append({
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name']
            })
        
        offset += 50
        if len(liked_songs) >= 5000:  # Limit to 5000 liked songs
            break
    
    return liked_songs

def prepare_dataset(csv_data, liked_songs):
    # Prepare liked songs
    liked_df = pd.DataFrame(liked_songs)
    liked_df['liked'] = 1
    
    # Merge liked songs with CSV data
    merged_liked = pd.merge(liked_df, csv_data, left_on='id', right_on='track_id', how='inner')
    
    # Sample random songs (not in liked songs)
    random_songs = csv_data[~csv_data['track_id'].isin(liked_df['id'])].sample(n=20000, random_state=42)
    random_songs['liked'] = 0
    
    # Combine datasets
    final_df = pd.concat([merged_liked, random_songs], ignore_index=True)
    
    return final_df