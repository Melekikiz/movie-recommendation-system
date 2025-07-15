import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


df = pd.read_csv("movie_ratings.csv")


rating_matrix = df.pivot_table(index="userId", columns="movieId", values="rating")
print("Rating Matrix:\n", rating_matrix)


rating_matrix_filled = rating_matrix.fillna(0)


user_similarity = cosine_similarity(rating_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
print("\nUser Similarity Matrix:\n", user_similarity_df)


target_user = int(input("Enter user ID to get movie recommendations: "))


def get_recommendations(target_user, similarity_df, rating_matrix, top_n=2):
    similar_users = similarity_df[target_user].drop(index=target_user).sort_values(ascending=False)
    top_users = similar_users.head(top_n).index
    similar_users_ratings = rating_matrix.loc[top_users]
    avg_ratings = similar_users_ratings.mean(axis=0)
    watched = rating_matrix.loc[target_user].dropna().index
    recommendations = avg_ratings.drop(index=watched, errors='ignore')
    return recommendations.sort_values(ascending=False).head(3)


recommended_movies = get_recommendations(target_user, user_similarity_df, rating_matrix)
print(f"\nRecommended movies for User {target_user}:\n", recommended_movies)


user_ratings = rating_matrix.loc[target_user].dropna()
user_ratings.plot(kind="bar", title=f"User {target_user}'s Ratings", color="skyblue")
plt.ylabel("Rating")
plt.show()


recommended_movies.plot(kind="bar", title=f"Recommended Movies for User {target_user}", color="lightgreen")
plt.ylabel("Predicted Rating")
plt.show()
