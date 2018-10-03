def baseline():
    root = os.path.join(os.path.dirname(__file__), "../")
    trainer = Trainer(root, preprocessor_name="test_train_preprocessor", log_dir="test")

    vocab_size = len(trainer.preprocessor.vocabulary.get())
    model = LSTMLM(vocab_size, embedding_size=100, hidden_size=50)

    metrics = trainer.train(model, data_kind="valid", epochs=1)
    last_acc = metrics.history["acc"][-1]
    shutil.rmtree(trainer.log_dir)
    os.remove(trainer.preprocessor_path)
    self.assertTrue(last_acc > 0.2)
