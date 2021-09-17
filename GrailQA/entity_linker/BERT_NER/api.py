"""
 Copyright 2021, Ohio State University (Yu Gu)
 Yu Gu  <gu.826@osu.edu>
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import Ner

app = Flask(__name__)
CORS(app)

model = Ner("out_!x")

@app.route("/predict",methods=['POST'])
def predict():
    text = request.json["text"]
    try:
        out = model.predict(text)
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000)