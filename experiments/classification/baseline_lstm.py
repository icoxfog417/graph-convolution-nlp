import os
import sys
import numpy as np
from sklearn.metrics import classification_report
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from gcn.data.multi_nli_dataset import MultiNLIDataset
from gcn.classification.baseline import LSTMClassifier
from gcn.classification.baseline_trainer import BaselineTrainer


def main():
    root = os.path.join(os.path.dirname(__file__), "../../")
    dataset = MultiNLIDataset(root)
    trainer = BaselineTrainer(root, log_dir="classifier_baseline")
    trainer.build()
    sequence_length = 25

    vocab_size = len(trainer.preprocessor.vocabulary.get())

    def preprocessor(x):
        _x = trainer.preprocess(x, sequence_length)
        return _x["text"]

    model = LSTMClassifier(vocab_size)
    model.build(trainer.num_classes, preprocessor)

    metrics = trainer.train(model.model, epochs=25,
                            sequence_length=sequence_length,
                            representation="GloVe.6B.100d")

    test_data = dataset.test_data()
    y_pred = model.predict(test_data["text"])

    print(classification_report(test_data["label"], y_pred,
                                target_names=dataset.labels()))


if __name__ == "__main__":
    main()
