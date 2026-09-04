import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Treina o modelo na inicialização com o dataset sintético
# Para usar dados reais: substitua pelo pd.read_csv("MKT.csv")
def _treinar_modelo() -> LinearRegression:
    rng = np.random.default_rng(42)
    n = 171
    youtube   = rng.uniform(0.7,  449.0, n)
    facebook  = rng.uniform(0.0,  300.0, n)
    newspaper = rng.uniform(0.3,  114.0, n)
    sales = (
        3.0 + 0.045 * youtube + 0.188 * facebook + 0.001 * newspaper
        + rng.normal(0, 1.5, n)
    ).clip(1.92, 32.4)

    data = pd.DataFrame({"youtube": youtube, "facebook": facebook, "newspaper": newspaper, "sales": sales})
    X = data[["youtube", "facebook", "newspaper"]]
    y = data["sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

_model = _treinar_modelo()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        youtube   = float(body["youtube"])
        facebook  = float(body["facebook"])
        newspaper = float(body["newspaper"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Parâmetros inválidos."}), 400

    entrada = pd.DataFrame([[youtube, facebook, newspaper]], columns=["youtube", "facebook", "newspaper"])
    sales = round(float(_model.predict(entrada)[0]), 2)
    return jsonify({"sales": sales})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
