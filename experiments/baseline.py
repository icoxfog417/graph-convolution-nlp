import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from gcn.language_model.trainer import Trainer
from gcn.language_model.baseline import LSTMLM


def main():
    root = os.path.join(os.path.dirname(__file__), "../")
    trainer = Trainer(root, train_data_limit=50000,
                      preprocessor_name="baseline_preprocessor", log_dir="baseline")
    trainer.build()
    vocab_size = len(trainer.preprocessor.vocabulary.get())
    print("vocab size: {}".format(vocab_size))
    model = LSTMLM(vocab_size, embedding_size=100, hidden_size=30)
    metrics = trainer.train(model, epochs=20)


if __name__ == "__main__":
    main()
