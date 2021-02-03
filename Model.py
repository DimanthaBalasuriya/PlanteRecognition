from keras.models import model_from_json
import numpy as np


class DiseaseDetectionModel(object):
    RICEDISEASELIST = ["Blast", "Blight", "Brownspot", "Sheath Blight", "Tungro"]

    def __init__(self, model_json_file, model_weight_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weight_file)
        self.loaded_model._make_predict_function()

    def predict_disease(self, img):
        self.preds = self.loaded_model.predict(img)
        return DiseaseDetectionModel.RICEDISEASELIST[np.argmax(self.preds)]
