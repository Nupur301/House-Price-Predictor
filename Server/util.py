import json
import pickle
import numpy as np


locations = None
data_columns = None
model = None

def get_location_names():
    return locations


def get_estimated_price(location, sqft, bhk, bath):
    loc_index = data_columns.index(location.lower())
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0] , 2)


def load_saved_artifacts():
    print("loading artifacts...START")
    global data_columns
    global locations
    global model
   
    with open("./artifacts/columns.json" , "r") as f:
        data_columns = json.load(f)["data_columns"]
        locations = data_columns[3:]

    with open("./artifacts/banglore_home_prices_model.pickle" , "rb") as f:
        model = pickle.load(f)
    print("loading ...DONE")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("1st phase jp nagar" , 1000, 3, 3))

