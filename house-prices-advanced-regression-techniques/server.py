
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request
from flasgger import Swagger

with open('rf.pkl', 'rb') as model_rfc_pickle:
    model_rfc = pickle.load(model_rfc_pickle)

ml_api = Flask(__name__)
swagger = Swagger(ml_api)

@ml_api.route('/predict', methods=['GET'])
def predict():
    """
    ---
    :parameters:
        - name: LotArea
          in: query
          type: number
          required: true
        - name: OverallQua
          in: query
          type: number
          required: true
        - name: OverallCond
          in: query
          type: number
          required: true
        - name: YearBuilt
          in: query
          type: number
          required: true
        - name: YearRemodAdd
          in: query
          type: number
          required: true
        - name: TotalBsmtSF
          in: query
          type: number
          required: true
        - name: SaleCondition_Family
          in: query
          type: number
          required: true
        - name: SaleCondition_Normal
          in: query
          type: number
          required: true
        - name: SaleCondition_Partial
          in: query
          type: number
          required: true
    """

    LotArea = request.args.get("LotArea")
    OverallQua = request.args.get("OverallQua")
    OverallCond = request.args.get("OverallCond")
    YearBuilt = request.args.get("YearBuilt")
    YearRemodAdd = request.args.get("YearRemodAdd")
    TotalBsmtSF = request.args.get("TotalBsmtSF")
    SaleCondition_Family = request.args.get("SaleCondition_Family")
    SaleCondition_Normal = request.args.get("SaleCondition_Normal")
    SaleCondition_Partial = request.args.get("SaleCondition_Partial")

    input_data = np.array([[LotArea, OverallQua, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, SaleCondition_Family, SaleCondition_Normal, SaleCondition_Partial]])
    prediction = model_rfc.predict(input_data)
    return str(prediction)


@ml_api.route('/predict_rfc', methods=["POST"])
def predict_rfc():
    """Endpoint to predict the house price using Random Forest Classifier using file input
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    """

    input_data = pd.read_csv(request.files.get('input_file'))
    prediction = model_rfc.predict(input_data)
    return str(list(prediction))


if __name__ == '__main__':
    ml_api.run(host='0.0.0.0', port=8888)



# from flask import Flask, jsonify, request
# import pandas as pd
# import joblib
#
# app = Flask(__name__)
#
# @app.route("/", methods=["POST"])
# def do_prediction():
#     json = request.get_json()
#     model = joblib.load('model.pkl')
#     df = pd.DataFrame(json, index=[0])
#
#     #from sklearn.ensemble import RandomForestClassifier
#     #from sklearn.ensemble import RandomForestRegressor
#     from sklearn.preprocessing import StandardScaler
#     scalar = StandardScaler()
#     scalar.fit(df)
#
#     df_x_scaled = scalar.transform(df)
#
#     df_x_scaled = pd.DataFrame(df_x_scaled, columns=df.columns)
#     y_predict = model.predict(df_x_scaled)
#
#     result = {"Predicted House Price": y_predict[0]}
#     return jsonify(result)
#
#
# if __name__ == "__main__":
#     app.run(host='0.0.0.0')
#
# import os
# from flask import jsonify, request
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
#
#
# os.system('clear')
# def do_prediction():
#     json = request.get_json()
#     model = joblib.load('model.pkl')
#     df = pd.DataFrame(json, index=[0])
#
#     scalar = StandardScaler()
#     scalar.fit(df)
#
#     df_x_scaled = scalar.transform(df)
#     df_x_scaled = pd.DataFrame(df_x_scaled, columns=df.columns)
#
#     y_predict = model.predict(df_x_scaled)
#
#     result = {"Predicted House Price": y_predict[0]}
#     print(result)
#     return jsonify(result)








