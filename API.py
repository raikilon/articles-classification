from Scraper import Scraper
from Classifier import Classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from flask import Flask
from flask_restful import Api, Resource
import numpy as np

app = Flask(__name__)
api = Api(app)


class Model(Resource):
    def get(self):
        # scrape
        scraper = Scraper()
        df = scraper.get()

        # Encode labels and split for training
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
        train, val = train_test_split(df, stratify=df.label, test_size=0.2)

        classifier = Classifier()

        # prepare data for BERT model
        train_dataloader = classifier.prepare_data(train)
        val_dataloader = classifier.prepare_data(val)

        best_acc = classifier.train(train_dataloader, val_dataloader)

        return {'best_acc': float(best_acc)}


class Evaluation(Resource):
    def get(self, id):
        scraper = Scraper()
        df = scraper.load()
        if not df.empty and 0 <= id < df.label.count():
            label_encoder = LabelEncoder()
            df['label'] = label_encoder.fit_transform(df['label'])

            classifier = Classifier(True)
            dataloader = classifier.prepare_data(df.iloc[[id]], 1)

            _, predictions, true_labels = classifier.evaluate(dataloader)
            pred = int(np.argmax(predictions, axis=1)[0])
            label = int(true_labels[0])
            return {'prediction': pred, 'true_label': label}

        return {'message': 'error'}


api.add_resource(Evaluation, "/classify/<int:id>")
api.add_resource(Model, "/train")

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
