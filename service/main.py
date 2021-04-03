from fastapi import FastAPI
from typing import Optional
from typing import List
from pydantic import BaseModel
import sklearn
from sklearn.ensemble import IsolationForest
import joblib

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None



class PredictionParams(BaseModel):
    feature_vector: List[float]
    score: Optional[bool] = False

class PredictionResponse(BaseModel):
    is_inlier: bool
    anomaly_score: Optional[float] = None

app = FastAPI()

joblib_file="anomalyModel.joblib"

joblib_model = joblib.load(joblib_file)


@app.get("/model_information")
async def root():
    return joblib_model.get_params()

@app.post("/prediction/")
async def prediction(params: PredictionParams):
    print(params.feature_vector)
    predicted = joblib_model.predict([params.feature_vector])
    response = PredictionResponse(is_inlier= int(predicted[0]))

    
    if params.score is True:
        score = joblib_model.score_samples([params.feature_vector])
        response.anomaly_score = score[0]

    return response
    #return predicted

@app.post("/items/")
async def create_item(item: Item):
    return item
