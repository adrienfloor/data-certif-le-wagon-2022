
# Data certification API

Le Wagon Data Science certification exam starter pack for the predictive API test.

**💡&nbsp;&nbsp;This challenge is completely independent of other challenges. It is not required to complete any other challenge in order to work on this challenge.**

## API challenge

**📝&nbsp;&nbsp;In this challenge, you are provided with a trained model saved as `assets/model.joblib`. The goal is to create an API that will predict the popularity of a song based on its other features.**

👉&nbsp;&nbsp;You will only need to edit the code of the API in `api/fast.py` 🚨

👉&nbsp;&nbsp;The package versions listed in `requirements.txt` should work out of the box with the pipelined model saved in `assets/model.joblib`

### Install the required packages

The `requirements.txt` file lists the exact version of the packages required in order to be able to load the pipelined model that we provide.

You may want to install these required package in a new, dedicated virtual environment.

``` bash
pip install -r requirements.txt
```

### Run a uvicorn server

**📝&nbsp;&nbsp;Start a `uvicorn` server in order to make sure that the setup works correctly.**

Run the server:

```bash
uvicorn api.fast:app --reload
```

Open your browser at http://localhost:8000/

👉&nbsp;&nbsp;You should see the response `{"Status": "Up and running"}`

You will now be able to work on the content of the API while `uvicorn` automatically reloads your code as it changes.

### API specification

**Predict the popularity of a Spotify song**

`GET /predict`

| Parameter | Type | Description |
|---|---|---|
| acousticness | float | whether the track is acoustic |
| danceability | float | describes how suitable a track is for dancing |
| duration_ms | int | duration of the track in milliseconds |
| energy | float | represents a perceptual measure of intensity and activity |
| explicit | int | whether the track has explicit lyrics |
| id | string | id for the track |
| instrumentalness | float | predicts whether a track contains no vocals |
| key | int | the key the track is in |
| liveness | float | detects the presence of an audience in the recording |
| loudness | float | the overall loudness of a track in decibels |
| mode | int | modality of a track |
| name | string | name of the track |
| release_date | string | release date |
| speechiness | float | detects the presence of spoken words in a track |
| tempo | float | overall estimated tempo of a track in beats per minute |
| valence | float | describes the musical positiveness conveyed by a track |
| artist | string | artist who performed the track |

Returns a dictionary with the `artist`, the `name` of the song and predicted `popularity` as an integer.

Example request:

```
/predict?acousticness=0.654&danceability=0.499&duration_ms=219827&energy=0.19&explicit=0&id=0B6BeEUd6UwFlbsHMQKjob&instrumentalness=0.00409&key=7&liveness=0.0898&loudness=-16.435&mode=1&name=Back%20in%20the%20Goodle%20Days&release_date=1971&speechiness=0.0454&tempo=149.46&valence=0.43&artist=John%20Hartford
```

Example response:

``` json
{
  "artist": "John Hartford",
  "name": "Back in the Goodle Days",
  "popularity": 22
}
```

👉 It is your turn, code the endpoint in `api/fast.py`. If you want to verify what data types the pipeline expects, have a look at the docstring of the `create_pipeline` method in `trainer/trainer.py`.

## API in production

**📝&nbsp;&nbsp;Push your API to production on the hosting service of your choice.**

<details>
  <summary>👉&nbsp;&nbsp;If you opt for Google Cloud Platform 👈</summary>

  &nbsp;


Once you have changed your `GCP_PROJECT_ID` in the `Makefile`, run the directives of the `Makefile` to build and deploy your containerized API to Container Registry and finally Cloud Run.

</details>
