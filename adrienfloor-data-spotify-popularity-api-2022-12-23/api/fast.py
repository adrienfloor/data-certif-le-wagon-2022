from datetime import datetime
import pytz
import pandas as pd
import joblib
import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

current_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../assets'))
loaded_model = joblib.load(f"{current_dir}/model.joblib")
app.state.model = loaded_model

# define a root `/` endpoint
@app.get("/")
def index():
    return {"Status": "Up and running"}

# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness: float,
            id: object,
            danceability: float,
            duration_ms: int,
            energy: float,
            explicit: int,
            instrumentalness: float,
            key: int,
            liveness: float,
            loudness: float,
            mode: int,
            name: object,
            release_date: object,
            speechiness: float,
            tempo: float,
            valence: float,
            artist: object):

    X_pred = pd.DataFrame(dict(
    acousticness=[acousticness],
    danceability=[danceability],
    duration_ms=[duration_ms],
    id=[id],
    energy=[energy],
    explicit=[explicit],
    instrumentalness=[instrumentalness],
    key=[key],
    liveness=[liveness],
    loudness=[loudness],
    mode=[mode],
    name=[name],
    release_date=[release_date],
    speechiness=[speechiness],
    tempo=[tempo],
    valence=[valence],
    artist=[artist]))

    model = app.state.model
    y_pred = model.predict(X_pred)

    print('')
    print('')
    print(f'This is the prediction : {y_pred}')
    print('')
    print('')

    # ⚠️ fastapi only accepts simple python data types as a return value
    # among which dict, list, str, int, float, bool
    # in order to be able to convert the api response to json
    return dict(Artist=str(artist),name=str(name),Popularity=int(y_pred))
