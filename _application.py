from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = FastAPI()

app = application

templates = Jinja2Templates(directory='templates') # Assuming your templates are in a 'templates' directory

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_datapoint(request: Request, 
                            carat: float = Form(...), 
                            depth: float = Form(...), 
                            table: float = Form(...), 
                            x: float = Form(...), 
                            y: float = Form(...), 
                            z: float = Form(...), 
                            cut: str = Form(...), 
                            color: str = Form(...), 
                            clarity: str = Form(...)):
    data = CustomData(carat, depth, table, x, y, z, cut, color, clarity)
    pred_df = data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(pred_df)
    results = round(pred[0], 2)

    return templates.TemplateResponse("index.html", 
                                      {"request": request, "results": results, "pred_df": pred_df})

@app.post("/predictAPI")
async def predict_api(
    carat: float, 
    depth: float, 
    table: float, 
    x: float, 
    y: float, 
    z: float, 
    cut: str, 
    color: str, 
    clarity: str):
    data = CustomData(carat, depth, table, x, y, z, cut, color, clarity)
    pred_df = data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(pred_df)

    return {"price": round(pred[0], 2)}

# FastAPI does not require the 'if __name__ == "__main__"' block to run; 
# it's run using a server like Uvicorn or Hypercorn
