require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake

README_FILE = 'README.md'.freeze

define_tasks('qnd', define_pytest: false)

task_in_venv :pytest do
  vsh 'cd examples/lib && . ./mnist.sh && fetch_dataset'

  Dir.glob(['qnd/**/*_test.py', 'examples/**/*_test.py']).each do |file|
    vsh :pytest, file
  end
end

task_in_venv :script_test do
  vsh 'python3 test/empty.py'

  Dir.glob('test/*.py').each do |file|
    vsh :python3, file, '-h'
  end

  distributed_oracle = 'distributed=yes python3 test/oracle.py'
  vsh "#{distributed_oracle} -h"

  # Worker hosts should not include a master host.
  vsh('!', distributed_oracle.to_s,
      '--master_host', 'localhost:4242',
      '--worker_hosts', 'localhost:4242',
      '--ps_hosts', 'localhost:5151',
      '--task_type', 'job',
      '--train_file', 'README.md',
      '--eval_file', 'setup.py')
end

%i[mnist_simple mnist_distributed mnist_evaluate mnist_infer].each do |name|
  task_in_venv name do
    vsh "cd examples/#{name} && ./main.sh"
  end
end

task_in_venv :mnist_full do |t|
  [
    nil,
    %i[use_eval_input_fn],
    %i[use_dict_inputs],
    %i[use_model_fn_ops],
    %i[self_batch],
    %i[self_filename_queue use_eval_input_fn]
  ].each do |flags|
    vsh(
      'cd', "examples/#{t.name}", '&&',
      (flags && flags.map { |flag| "#{flag}=yes" }.join(' ')).to_s, './main.sh'
    )
  end
end

task test: %i[
  pytest
  script_test
  mnist_simple
  mnist_distributed
  mnist_evaluate
  mnist_infer
  mnist_full
]

task :readme_examples do
  md = File.read(README_FILE)

  command_script = 'train.py'
  library_script = 'mnist.py'

  def read_example_file(file)
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

task doc: %i[pdoc readme_examples]
