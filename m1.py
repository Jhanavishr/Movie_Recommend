from flask import Flask, render_template, request
from flask_mysqldb import MySQL

import numpy as np
import pandas as pd
import pickle


# Assuming new_df is your DataFrame containing movie titles
#movies_list = new_df['title'].values

# Pickle the movies list
#with open('movies.pkl', 'wb') as f:
    #pickle.dump(movies_list, f)


try:
    with open('movies.pkl', 'rb') as f:
        movies_list = pickle.load(f)
    print("Movies loaded successfully!")
except FileNotFoundError:
    print("The pickled file 'movies.pkl' does not exist.")
except Exception as e:
    print("An error occurred while loading the pickled file:", e)



app = Flask(__name__)
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='jhanavi2904'
app.config['MYSQL_DB']='movie'
mysql = MySQL(app)

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies.merge(movies[['title', 'release_date']], on='title')
# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date_y']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
movies.drop_duplicates(inplace=True)

movies.duplicated().sum()
import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
import ast
def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movies['cast']=movies['cast'].apply(convert3)
def find_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
movies['crew']=movies['crew'].apply(find_director)
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['release_year'] = movies['release_date_y'].apply(lambda x: x.split('-')[0])

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
new_df=movies[['movie_id','title','tags','overview','genres','release_year']]
#list to string
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())

import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)
new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
vectors = cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)

import requests
def recommend(movie):
    # Find the index of the given movie
    movie_index = new_df[new_df['title'] == movie].index
    if len(movie_index) == 0:
        print(f"Movie '{movie}' not found.")
        return [], []  # Return empty lists for both titles and IDs

    movie_index = movie_index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_titles = []
    recommended_ids = []
    for i in movies_list:
        recommended_movie_title = new_df.iloc[i[0]].title
        recommended_movie_id = new_df.iloc[i[0]].movie_id
        print(f"Recommended Movie: {recommended_movie_title}, Movie ID: {recommended_movie_id}")
        recommended_titles.append(recommended_movie_title)
        recommended_ids.append(recommended_movie_id)
    return recommended_titles, recommended_ids

def overviews(movie):
    # Find the index of the given movie
    movie_index = new_df[new_df['title'] == movie].index
    if len(movie_index) == 0:
        print(f"Movie '{movie}' not found.")
        return [], []  # Return empty lists for both titles and IDs

    movie_index = movie_index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    movie_overviews = []
    for i in movies_list:
        movie_overview = new_df.iloc[i[0]].overview
        movie_overviews.append(" ".join(movie_overview))
    return movie_overviews

def fetch_posters(movie_ids):
    posters = {}  # Dictionary to store movie names and their poster URLs
    for movie_id in movie_ids:
        print(f"Fetching poster for Movie ID: {movie_id}")
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=e18747b0d4f206805bc750c03d860762"
        try:
            response = requests.get(url)
            #response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404, 500)
            data = response.json()
            movie_name = data.get('title')
            poster_path = data.get('poster_path')
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                print("Poster URL:", poster_url)
                posters[movie_name] = poster_url  # Store movie name and poster URL in the dictionary
            else:
                print("Poster path not found in API response.")
        except requests.exceptions.RequestException as e:
            print("Error fetching poster:", e)
    return posters




movies['release_year'] = movies['release_year'].astype(int)
"""
def recommend_year():
    # Ask the user for the release year
    year = input("Enter the release year: ")
    
    # Convert the input to integer
    year = int(year)
    
    # Filter movies based on release year
    filtered_movies = movies[movies['release_year'] == year].copy()
    
    # Concatenate 'title' and 'overview' columns with a newline character
    filtered_movies['title_overview'] = filtered_movies['title'] + ': ' + filtered_movies['overview'].apply(lambda x: ' '.join(x))
    
    # Print the concatenated titles and overviews
    for title_overview in filtered_movies['title_overview'][0:5]:
        print(title_overview + '\n')

# Call the function to filter movies based on user input and print the result
recommend_year()"""

new_df.loc[:, 'genres'] = new_df['genres'].apply(lambda x: ' '.join(x))
g_vector = cv.fit_transform(new_df['genres']).toarray()

"""

def recommend_by_genre(movie_title, num_recommendations=5):
    # Find the index of the given movie
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    
    # Get the similarity scores for the given movie
    movie_similarities = similarity[movie_index]
    
    # Sort the movies based on similarity scores
    similar_movies_indices = movie_similarities.argsort()[::-1][1:num_recommendations+1]
    
    # Get the titles and overviews of recommended movies
    recommended_movies = []
    for index in similar_movies_indices:
        recommended_movies.append((new_df.iloc[index]['title'], new_df.iloc[index]['overview']))
    
    return recommended_movies

# Example: Recommend movies similar to 'Avatar' based on genres

recommendations = recommend_by_genre('Avatar')

print("Recommended movies based on genres similar to 'Avatar':")
for title, overview in recommendations:
    print(f"Title: {title}\nOverview: {' '.join(overview)}\n")
"""
def recommend_year():
    year = input("Enter the release year: ")

    year = int(year)
    
    # Filter movies based on release year
    filtered_movies = movies[movies['release_year'] == year].copy()
    
    # Concatenate 'title' and 'overview' columns with a newline character
    titles_and_overviews = filtered_movies.apply(lambda row: f"{row['title']}: {' '.join(row['overview'])}\n", axis=1)
    
    # Join the titles and overviews into a single string
    recommendations = ''.join(titles_and_overviews[0:5])
    
    return recommendations





@app.route('/')
def main_menu():
    return render_template('mainmenu.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Fetch user data from the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE Username = %s", (username,))
        user_data = cur.fetchone()
        cur.close()

        if user_data is None:
        
            return render_template('login.html', error='Invalid username or password')

        
        stored_password = user_data[-1]

        if password == stored_password:
            
            return render_template('mainmenu.html')
        else:
        
            return render_template('login.html', error='Invalid username or password')


    return render_template('login.html')







@app.route('/logout')
def logout():
    return render_template('logout.html')




@app.route('/log')
def log():
    return render_template('log.css')






#@app.route('/recommend_by_movie', methods=['GET', 'POST'])
@app.route('/recommend_by_movie', methods=['GET', 'POST'])
def recommend_by_movie():
    if request.method == 'POST':
        selected_movie = request.form.get('movieSelect')
        recommended_movies = recommend(selected_movie)
        movie_overview = overviews(selected_movie)
        recommended_posters = [fetch_posters(movie_id) for movie_id in recommended_movies]  # Fetch posters for recommended movies
        print(recommended_movies)
        
        for movie, poster_url in zip(recommended_movies, recommended_posters):
            print(f"Movie: {movie}, Poster URL: {poster_url}")

        recommended_data = zip(recommended_movies, movie_overview, recommended_posters)
        print(list(recommended_data))
        return render_template('recommend_by_movie.html', selected_movie=selected_movie, poster_url=poster_url, overview=movie_overview)
     
    return render_template('recommend_by_movie.html', movies=movies_list)
unique_release_years = sorted(new_df['release_year'].unique().tolist())

@app.route('/recommend_by_releaseyear', methods=['POST','GET'])
def recommend_by_year():
    if request.method == 'POST':
        selected_year = request.form.get('yearSelect')
        selected_year = int(selected_year)
        
        # Filter movies based on the selected year
        filtered_movies = movies[movies['release_year'] == selected_year].copy()

        # Get the titles and IDs of movies from the filtered list
        movie_titles = filtered_movies['title'].tolist()
        movie_ids = filtered_movies['movie_id'].tolist()

        # Fetch posters for recommended movies
        recommended_posters = fetch_posters(movie_ids)

        # Zip recommended movies with their posters
        recommended_data = zip(movie_titles, recommended_posters.values())
        

        
        return render_template('recommend_by_releaseyear.html', recommended_data=recommended_data)

    # Extract unique release years for the dropdown menu
    unique_release_years = sorted(new_df['release_year'].unique().tolist())

    
    return render_template('recommend_by_releaseyear.html', release_years=unique_release_years)

if __name__ == '__main__':
    app.run(debug=True,port=5501)
