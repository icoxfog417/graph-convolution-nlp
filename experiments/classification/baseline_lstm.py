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
    length = 5

    vocab_size = len(trainer.preprocessor.vocabulary.get())
    model = LSTMClassifier(vocab_size)
    model.build(trainer.num_classes)

    metrics = trainer.train(model.model, epochs=10,
                            sequence_length=length)

    test_data = trainer.preprocess(dataset.test_data(), length)
    y_pred = model.model.predict(test_data["text"])

    print(classification_report(test_data["label"], np.argmax(y_pred, axis=1),
                                target_names=dataset.labels()))


if __name__ == "__main__":
    main()
