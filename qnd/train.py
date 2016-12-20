from .experiment import def_experiment



def def_train_and_evaluate():
  experiment = def_experiemnt()

  def train_and_evaluate(model_fn, file_reader):
    experiment().train_and_evaluate()

  return train_and_evaluate
