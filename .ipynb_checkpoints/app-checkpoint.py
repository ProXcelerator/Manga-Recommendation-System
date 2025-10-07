{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "916c32f4-fdd4-4581-ab78-1ddda2d7fdd2",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'interaction_matrix.pkl'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[4], line 15\u001b[0m\n\u001b[0;32m      9\u001b[0m application \u001b[38;5;241m=\u001b[39m Flask(\u001b[38;5;18m__name__\u001b[39m)\n\u001b[0;32m     11\u001b[0m \u001b[38;5;66;03m# ---------------------------------------------------------------------------------------------------------------------#\u001b[39;00m\n\u001b[0;32m     12\u001b[0m \u001b[38;5;66;03m# -------------------------------------------- Load Interaction Matrix -----------------------------------------------#\u001b[39;00m\n\u001b[0;32m     13\u001b[0m \u001b[38;5;66;03m# ---------------------------------------------------------------------------------------------------------------------#\u001b[39;00m\n\u001b[0;32m     14\u001b[0m \u001b[38;5;66;03m# Load the saved interaction matrix from pickle file\u001b[39;00m\n\u001b[1;32m---> 15\u001b[0m interaction_matrix \u001b[38;5;241m=\u001b[39m joblib\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124minteraction_matrix.pkl\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m     16\u001b[0m interaction_matrix_values \u001b[38;5;241m=\u001b[39m interaction_matrix\u001b[38;5;241m.\u001b[39mvalues\n\u001b[0;32m     18\u001b[0m \u001b[38;5;66;03m# ---------------------------------------------------------------------------------------------------------------------#\u001b[39;00m\n\u001b[0;32m     19\u001b[0m \u001b[38;5;66;03m# ---------------------------------------------- Load Model ----------------------------------------------------------#\u001b[39;00m\n\u001b[0;32m     20\u001b[0m \u001b[38;5;66;03m# ---------------------------------------------------------------------------------------------------------------------#\u001b[39;00m\n\u001b[0;32m     21\u001b[0m \u001b[38;5;66;03m# Load the pre-trained model from pickle file\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\joblib\\numpy_pickle.py:650\u001b[0m, in \u001b[0;36mload\u001b[1;34m(filename, mmap_mode)\u001b[0m\n\u001b[0;32m    648\u001b[0m         obj \u001b[38;5;241m=\u001b[39m _unpickle(fobj)\n\u001b[0;32m    649\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 650\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mopen\u001b[39m(filename, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrb\u001b[39m\u001b[38;5;124m'\u001b[39m) \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[0;32m    651\u001b[0m         \u001b[38;5;28;01mwith\u001b[39;00m _read_fileobject(f, filename, mmap_mode) \u001b[38;5;28;01mas\u001b[39;00m fobj:\n\u001b[0;32m    652\u001b[0m             \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(fobj, \u001b[38;5;28mstr\u001b[39m):\n\u001b[0;32m    653\u001b[0m                 \u001b[38;5;66;03m# if the returned file object is a string, this means we\u001b[39;00m\n\u001b[0;32m    654\u001b[0m                 \u001b[38;5;66;03m# try to load a pickle file generated with an version of\u001b[39;00m\n\u001b[0;32m    655\u001b[0m                 \u001b[38;5;66;03m# Joblib so we load it with joblib compatibility function.\u001b[39;00m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'interaction_matrix.pkl'"
     ]
    }
   ],
   "source": [
    "# save this as app.py\n",
    "from flask import Flask, request, render_template\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "# ---------------------------------------------- Flask Setup ----------------------------------------------------------#\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "application = Flask(__name__)\n",
    "\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "# -------------------------------------------- Load Interaction Matrix -----------------------------------------------#\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "# Load the saved interaction matrix from pickle file\n",
    "interaction_matrix = joblib.load('interaction_matrix.pkl')\n",
    "interaction_matrix_values = interaction_matrix.values\n",
    "\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "# ---------------------------------------------- Load Model ----------------------------------------------------------#\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "# Load the pre-trained model from pickle file\n",
    "svd = joblib.load('svd_model.pkl')\n",
    "latent_matrix_train = svd.transform(interaction_matrix_values)\n",
    "latent_matrix_item = svd.components_\n",
    "\n",
    "# Recommendation function using collaborative filtering\n",
    "def recommend_collaborative(user_id, n=5):\n",
    "    if user_id not in interaction_matrix.index:\n",
    "        return [\"No user data available for recommendations.\"]\n",
    "    \n",
    "    user_index = interaction_matrix.index.get_loc(user_id)\n",
    "    if user_index >= latent_matrix_train.shape[0]:  # Handle cases where user may not be in the training set\n",
    "        return [\"User not in the training set for recommendations.\"]\n",
    "    \n",
    "    user_vector = latent_matrix_train[user_index]\n",
    "    \n",
    "    # Calculate similarity scores with all items\n",
    "    scores = np.dot(user_vector, latent_matrix_item)\n",
    "    \n",
    "    # Rank manga based on scores\n",
    "    recommended_books = np.argsort(scores)[::-1][:n]\n",
    "    recommended_book_titles = interaction_matrix.columns[recommended_books]\n",
    "    \n",
    "    return recommended_book_titles.tolist()\n",
    "\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "# -------------------------------------------- Flask Routes ----------------------------------------------------------#\n",
    "# ---------------------------------------------------------------------------------------------------------------------#\n",
    "@application.route('/')\n",
    "@application.route('/about')\n",
    "def about():\n",
    "    return render_template(\"about.html\")\n",
    "\n",
    "@application.route('/mangaRecommendation', methods=['GET', 'POST'])\n",
    "def mangaRecommendation():\n",
    "    recommendations = None\n",
    "    if request.method == \"POST\":\n",
    "        user_id = request.form.get('user_id')\n",
    "        try:\n",
    "            # Generate recommendations using the collaborative filtering model\n",
    "            recommendations = recommend_collaborative(user_id, n=5)\n",
    "        except ValueError:\n",
    "            recommendations = [\"Please enter valid values\"]\n",
    "    return render_template(\"manga_recommendation.html\", recommendations=recommendations)\n",
    "\n",
    "# Run on Correct Port\n",
    "if __name__ == '__main__':\n",
    "    application.debug = True\n",
    "    application.run(host=\"localhost\", port=5000, debug=True)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
