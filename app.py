from flask import Flask, render_template, request
from neural_model import predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            inp = [
                float(request.form.get('input0', -9999)),
                float(request.form.get('input1', -9999)),
                float(request.form.get('input2', -9999)),
                float(request.form.get('input3', -9999)),
            ]
            result = predict(inp)
        except Exception as e:
            result = f"Ошибка: {e}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)