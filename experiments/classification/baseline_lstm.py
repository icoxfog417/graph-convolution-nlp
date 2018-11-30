import os
import sys
import numpy as np
from sklearn.metrics import classification_report
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from gcn.data.multi_nli_dataset import MultiNLIDataset
from gcn.classification.trainer import Trainer
from gcn.classification.baseline import LSTMClassifier


def main():
    root = os.path.join(os.path.dirname(__file__), "../../")
    dataset = MultiNLIDataset(root)
    trainer = Trainer(root, preprocessor_name="c_preprocessor_baseline")
    trainer.build()
    sequence_length = 25

    vocab_size = len(trainer.preprocessor.vocabulary.get())

    def preprocessor(x):
        _x = trainer.preprocess(x, sequence_length)
        index = list(_x.keys())[0]
        return _x[index]

    model = LSTMClassifier(vocab_size)
    model.build(trainer.num_classes, preprocessor)

    metrics = trainer.train(model.model, epochs=25,
                            sequence_length=sequence_length)

    test_data = dataset.test_data()
    y_pred = model.predict(test_data["text"])

    print(classification_report(test_data["label"], y_pred,
                                target_names=dataset.labels()))


if __name__ == "__main__":
    main()
