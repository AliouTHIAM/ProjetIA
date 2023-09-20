# Importation des bibliothèques nécessaires
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
import pandas as pd


app = FastAPI()
@app.get("/predict")
async def predict(height:float,weight:float):
    # Chargement des données
    data= pd.read_csv("./data.csv")
    X = data[["Height","Weight"]]
    y = data["Sex"]

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialisation et entraînement du modèle d'arbre de décision
    dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
    dt.fit(X_train, y_train)
    
    # Prédiction sur des données de test
    new_data = [[height,weight]]
    predictions = dt.predict(new_data)
    # Calcul du gain
    gain = dt.score(X_test, y_test)

    return {"resultat":predictions[0],"gain":gain}
 


