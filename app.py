from flask import Flask, render_template, request, jsonify, send_file
import joblib
import pandas as pd

app = Flask(__name__)

df = pd.read_csv('oversampled_data.csv')

model = joblib.load('Credit_card_model')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['GET'])
def predict():
    row_number = int(request.args.get('row', 0))

    data_row = df.iloc[row_number - 1] if 0 < row_number <= len(df) else None

    if data_row is not None:
        values = [float(data_row[f'V{i + 1}']) for i in range(28)]
        amount = float(data_row['Amount'])

        y_pred = model.predict([values + [amount]])

        return jsonify({'result': int(y_pred[0])})
    else:
        return jsonify({'error': 'Invalid row number'})

@app.route('/download_csv')
def download_csv():
    return send_file('oversampled_data.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
