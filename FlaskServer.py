import flask
import werkzeug
import cv2
import tensorflow as tf

#Project plante

CATEGORIES = ["blast", "blight", "Brownspot", "sheath_blight", "tungro"]

app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    imageFile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imageFile.filename)
    print("\n Received File name : " + imageFile.filename)
    imageFile.save(filename)

    model = tf.keras.models.load_model("CNN.model")
    print(imageFile.filename)
    image = prepare(imageFile.filename)
    prediction = model.predict([image])
    prediction = list(prediction[0])
    print(CATEGORIES[prediction.index(max(prediction))])
    return str(CATEGORIES[prediction.index(max(prediction))])


def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


app.run(host="127.0.0.1", port=5000, debug=True)
