import pickle
from sklearn.ensemble import RandomForestRegressor
from preprocessing import load_data, preprocess_data

def train_model(data_path, model_path="house_price_model.pkl"):
    df = load_data(data_path)
    df = preprocess_data(df)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    model = RandomForestRegressor()
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Model saved to", model_path)
