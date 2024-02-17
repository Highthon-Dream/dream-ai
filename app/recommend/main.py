#! /usr/bin/env python
import numpy as np
from scipy.sparse.linalg import svds
import torch.nn.functional as F

from ..db.client import MySQLClient
from ..types import UserItem, UserOject, QuestinoTemplate, BucketListObject
from ..nlp.embedding import load_embedding_model, EmbeddingModel

def parsing_db(raw_data) -> UserItem:
    print(raw_data)
    return

def fetch_bucketlist(client: MySQLClient):
    result = client.execute_query("SELECT * FROM bucket_list")

    bucketlists = []
    for (created_at, id, user_id, content, title) in result:
        bucketlist = BucketListObject(created_at=created_at, id=id, user_id=user_id, content=content, title=title)
        bucketlists.append(bucketlist)
    return bucketlists

def fetch_questions(client: MySQLClient):
    result = client.execute_query("SELECT * FROM question")
    questions = []
    
    for (created_at, id, user_id, content, title) in result:
        question = QuestinoTemplate(created_at=created_at, id=id, user_id=user_id, content=content, title=title)
        questions.append(question)
    return questions

def fetch_users(client: MySQLClient):
    result = client.execute_query("SELECT * FROM user")
    # users = parsing_db(result)
    users = []
    for (created_at, id, identify, name) in result:
        user = UserOject(created_at=created_at, id=id, identify=identify, name=name)
        users.append(user)
    return users

def matrix_factorization(user_item_matrix: np.array):
    # Perform Singular Value Decomposition (SVD)
    # Here, k is the number of latent factors (adjust according to your data)
    U, sigma, Vt = svds(user_item_matrix, k=4)
    sigma = np.diag(sigma)
    user_item_predicted = np.dot(np.dot(U, sigma), Vt)

    print(user_item_predicted.shape)
    rating = np.argsort(-user_item_predicted, axis=1)[:, 1:]

    return rating

# def emb_user(users_info: ):

def update_recommendation_info(client: MySQLClient):
    users_info = fetch_users(client)

    if len(users_info) < 2:
        print("Required more than 2 users for recommendation.")
        return

    rating_results = matrix_factorization(users_info)

    [users_info for rating in rating_results]
    # parsing the result?

class RecommendationSystem():
    def __init__(self, client: MySQLClient):
        self.client = client
        self.emb_model: EmbeddingModel = load_embedding_model()
        self.load_user()

    def recommend_user(self, user_id, data):
        # title - content => get embedding
        post_data = {user.id: "" for user in self.users}
        for post in data:
            title, content = post.title, post.content
            writer_id = post.user_id
            post_data[writer_id] += f"{title}: {content}\n"

        posts = [item for (_, item) in post_data.items()]
        embs = self.emb_model.get_embs(posts)
        keys = list(post_data.keys())
        idx = keys.index(user_id)
        # print(embs.shape)
        # ranking = matrix_factorization(embs)[idx]
        # print(ranking)
        sim = F.cosine_similarity(embs[idx], embs).detach().numpy()
        sim[idx] = 0
        ranking = np.argsort(-sim)[:-1]
        ranked_user = [self.users[idx] for idx in ranking]
        return ranked_user
    
    def load_user(self):
        self.users = fetch_users(client=self.client)


if __name__ == "__main__":
    user_item_matrix = np.array([
        [5, 0, 3, 0, 4, 0],
        [0, 4, 0, 0, 0, 2],
        [3, 0, 4, 0, 3, 0],
        [0, 0, 0, 4, 0, 4],
        [0, 3, 0, 5, 0, 5]
    ], dtype=np.float32)

    print(matrix_factorization(user_item_matrix))
