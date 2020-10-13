
"""
Created on Sun June 28 20:06:35 2020

@author: alketcecaj
"""

from keras.models import load_model
from flask import Flask, request
from flasgger import Swagger
from flask import jsonify, make_response
import numpy as np
import pandas as pd
import json

Aeroporto = load_model('./MLP_Aerporto.h5')
Dozza = load_model('./MLP_Dozza.h5')
FICO = load_model('./MLP_FICO.h5')
MuseoArteModerna = load_model('./MLP_MuseoArteModerna.h5')
Pinacoteca = load_model('./MLP_Pinacoteca.h5')
models = {'Aeroporto':Aeroporto, 'Dozza':Dozza, 'FICO':FICO, 'MuseoArteModerna':MuseoArteModerna, 'Pinacoteca':Pinacoteca}

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/mymodel/<modelId>', methods=['POST'])
def execute_prediction_from_post(modelId):
    """Model dinamico
    ---
    parameters:
      - name: modelId
        in: path
        type: string
        required: true
      - in: body
        name: body

    responses:
        200:
           description: The output values

    """

    jsonBody = request.json
    #print('ModelId, Body', str(modelId),  str(jsonBody))
    input_data = list(jsonBody)
    array_data = np.array(input_data)
    model = models.get(modelId)
    prediction = model.predict(np.array([array_data]))
    prediction = prediction.tolist()
    res = str(prediction)
    return jsonify({'msg': res}), 200


'''@app.route('/model/<modelId>', methods=['POST'])
def execute_prediction(modelId):
    """Model dinamico
    ---
    parameters:
      - name: modelId
        in: path
        type: string
        required: true
      - in: body
        name: body

    responses:
        200:
           description: The output values

    """


    jsonBody = request.json
    print('ModelId, Body', str(modelId),  str(jsonBody))
    #print("ModelId {} -> Body: {}".format(modelId, jsonBody))
    return jsonify({'msg': "Ok"}), 200


@app.route('/pinacoteca')
def predict_pinacoteca():
    """Multistep forecasting with MLP for Pinacoteca Museum
    ---
    parameters:
      - name: p
        in: query
        type: number
        required: true
      - name: s
        in: query
        type: number
        required: true
      - name: t
        in: query
        type: number
        required: true
      - name: q
        in: query
        type: number
        required: true
      - name: qt
        in: query
        type: number
        required: true
      - name: st
        in: query
        type: number
        required: true
    responses:
        200:
           description: The output values

    """
    p = int(request.args.get("p"))
    s = int(request.args.get("s"))
    t = int(request.args.get("t"))
    q = int(request.args.get("q"))
    qt = int(request.args.get("qt"))
    st = int(request.args.get("st"))
    prediction = mlp_Pinacoteca.predict(np.array([[p, s, t, q, qt, st]]))
    prediction = prediction.tolist()
    res = json.dumps(prediction)
    return res
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0')
