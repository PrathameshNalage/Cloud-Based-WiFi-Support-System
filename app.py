from flask import Flask, render_template, request, jsonify
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and the vectorizer
svm_model = joblib.load('svm_wifi_troubleshooting_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Function to predict solution
def predict_solution(query, device, os):
    query_combined = query + ' ' + device + ' ' + os
    query_tfidf = tfidf.transform([query_combined])
    return svm_model.predict(query_tfidf)[0]

# Home route to render HTML chatbot UI
@app.route('/')
def home():
    return render_template('chatbot.html')

# Route to handle chatbot conversation
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    # Extract user input
    query_input = data.get('query')
    device_input = data.get('device')
    os_input = data.get('os')
    
    # Check if all inputs are received
    if query_input and device_input and os_input:
        # Predict solution based on inputs
        solution = predict_solution(query_input, device_input, os_input)
        return jsonify({"response": solution})
    else:
        return jsonify({"response": "Please provide all necessary details: query, device, and OS."})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
