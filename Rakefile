README_FILE = 'README.md'
VENV_DIR = '.venv'
IN_VENV = ". #{VENV_DIR}/bin/activate &&"


def vsh *args, **kwargs
  sh([IN_VENV, *args.map{ |x| x.to_s }].join(' '), **kwargs)
end


def task_in_venv name, &block
  task name => %i(clean venv) do |t|
    block.call t
  end
end


task :venv do
  sh "python3 -m venv #{VENV_DIR}" unless File.directory? VENV_DIR

  vsh "pip3 install --upgrade #{%w(
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0-cp35-cp35m-linux_x86_64.whl
    pytest
    pdoc
    autopep8
  ).join ' '}"

  vsh 'python3 setup.py install'
end


task :clean do
  sh 'git clean -dfx'
end


task_in_venv :module_test do
  Dir.glob('qnd/**/*_test.py').each do |file|
    vsh :pytest, file
  end
end


task_in_venv :script_test do
  vsh('python3 test/empty.py')

  Dir.glob('test/*.py').each do |file|
    vsh :python3, file, '-h'
  end

  distributed_oracle = 'distributed=yes python3 test/oracle.py'
  vsh "#{distributed_oracle} -h"

  # Worker hosts should not include a master host.
  vsh(*%W(! #{distributed_oracle}
          --master_host localhost:4242
          --worker_host localhost:4242
          --ps_hosts localhost:5151
          --task_type job
          --train_file README.md
          --eval_file setup.py))
end


%i(mnist_simple mnist_distributed mnist_infer).each do |name|
  task_in_venv name do
    vsh "cd examples/#{name} && ./main.sh"
  end
end


task_in_venv :mnist_full do |t|
  [
    nil,
    %i(use_eval_input_fn),
    %i(use_dict_inputs),
    %i(use_model_fn_ops),
    %i(self_batch),
    %i(self_filename_queue use_eval_input_fn),
  ].each do |flags|
    vsh(*%W(cd examples/#{t.name} &&
            #{flags and flags.map{ |flag| "#{flag}=yes"}.join(' ')} ./main.sh))
  end
end


task :test => %i(
  module_test
  script_test
  mnist_simple
  mnist_distributed
  mnist_infer
  mnist_full
)


task :readme_examples do
  md = File.read(README_FILE)

  command_script = 'train.py'
  library_script = 'mnist.py'

  def read_example_file file
    File.read(File.join('examples/mnist_simple', file)).strip
  end

  File.write(README_FILE, %(
#{md.match(/(\A.*## Examples)/m)[0]}

`#{command_script}` (command script):

```python
#{read_example_file command_script}
```

`#{library_script}` (module):

```python
#{read_example_file library_script}
```

With the code above, you can create a command with the following interface.

```
#{`#{IN_VENV} cd examples/mnist_simple && python3 #{command_script} -h`.strip}
```

Explore [examples](examples) directory for more information and see how to run
them.


#{md.match(/## Caveats.*\Z/m)[0].strip}
).lstrip)
end


task_in_venv :html do
  vsh 'pdoc --html --html-dir docs --overwrite qnd'
end


task :doc => %i(html readme_examples)


task_in_venv :format do
  sh "autopep8 -i #{Dir.glob('**/*.py').join ' '}"
end


task :upload => %i(test clean) do
  sh 'python3 setup.py sdist bdist_wheel'
  sh 'twine upload dist/*'
end
