import tensorflow as tf, sys
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import base64

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.values['imageBase64']
    with open("imageToPredict.jpeg", "wb") as fh:
        fh.write(base64.standard_b64decode(data))

    image_path = 'imageToPredict.jpeg'

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("data/output_labels.txt")]

    with tf.gfile.FastGFile("data/output_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        output_string = []
        output_score = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            output_string.append(human_string)
            output_score.append(score)
            print('%s (score = %.5f)' % (human_string, score))
    return jsonify(results=[output_string])

@app.route('/')
def main():
    return render_template('index.html')