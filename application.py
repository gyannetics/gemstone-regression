from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
CORS(application)  # Enable CORS for the entire app

@app.route('/')
@cross_origin()
def home_page():
    """ Render the home page. """
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    """
    Endpoint to predict data points.
    GET returns the prediction form.
    POST processes the form and returns predictions.
    """
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Retrieve form data and create a CustomData object
        data = CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )

        # Generate a DataFrame and make a prediction
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = round(pred[0], 2)

        return render_template('index.html', results=results, pred_df=pred_df)
    
@app.route('/predictAPI', methods=['POST'])
@cross_origin()
def predict_api():
    """
    API endpoint for prediction.
    Expects JSON data and returns prediction in JSON format.
    """
    if request.method == 'POST':
        data = CustomData(
            carat=float(request.json['carat']),
            depth=float(request.json['depth']),
            table=float(request.json['table']),
            x=float(request.json['x']),
            y=float(request.json['y']),
            z=float(request.json['z']),
            cut=request.json['cut'],
            color=request.json['color'],
            clarity=request.json['clarity']
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        dct = {'price': round(pred[0], 2)}
        return jsonify(dct)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
