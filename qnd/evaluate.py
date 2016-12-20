from .experiment import def_experiment



def def_evaluate():
  experiment = def_experiment()

  def evaluate(model_fn, file_reader):
    experiemnt(model_fn, file_reader).evaluate()

  return evaluate
