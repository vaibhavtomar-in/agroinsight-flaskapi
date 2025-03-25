from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
#test
import os

#test
# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scalers
# model = pickle.load(open('model.pkl', 'rb'))
# sc = pickle.load(open('standscaler.pkl', 'rb'))
# ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

#test
# Load the trained model and scalers with updated paths
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(BASE_DIR, 'standscaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(BASE_DIR, 'minmaxscaler.pkl'), 'rb'))



# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = "Content-Type"


# Define crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route("/")
def home():
    return "Welcome to the Crop Recommendation API!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get input data
        data = request.json  # Expecting JSON input

        N = int(data['Nitrogen'])
        P = int(data['Phosporus'])
        K = int(data['Potassium'])
        temp = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['Ph'])
        rainfall = float(data['Rainfall'])

        # Prepare input features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply transformations
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Predict the crop
        prediction = model.predict(final_features)[0]

        # Get crop name from dictionary
        crop = crop_dict.get(prediction, "Unknown crop")
        print(crop)

        # Return response as JSON
        return jsonify({"Recommended Crop": crop})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
