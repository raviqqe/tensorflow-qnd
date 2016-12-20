from .experiment import def_experiment



def def_train_and_evaluate():
  experiment = def_experiment()

  def train_and_evaluate(model_fn, input_fn):
    experiment(model_fn, input_fn).train_and_evaluate()

  return train_and_evaluate
