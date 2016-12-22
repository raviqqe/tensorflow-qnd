VENV_DIR = '.venv'
TENSORFLOW_URL = 'https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp35-cp35m-linux_x86_64.whl'


def task_in_venv name, packages=['gargparse', TENSORFLOW_URL], &block
  task name => %i(clean venv) do
    def vsh *args, **kwargs
      sh([". #{VENV_DIR}/bin/activate &&", *args.map{ |x| x.to_s }].join(' '),
         **kwargs)
    end

    vsh 'pip3 install --upgrade', *packages
    block.call
  end
end


task :venv do
  sh "python3 -m venv #{VENV_DIR}" unless File.directory? VENV_DIR
end


task :clean do
  sh 'git clean -dfx'
end


task_in_venv :module_test do
  `find .`.split.select do |file|
    file =~ /_test\.py$/ and file !~ /\/\./
  end.each do |file|
    vsh 'pytest', file
  end
end


task_in_venv :mnist_example, [TENSORFLOW_URL] do
  ['python3 setup.py install', 'make -C examples/mnist'].each do |command|
    vsh command
  end
end


task :test => %i(module_test mnist_example)


task :upload => %i(test clean) do
  sh 'python3 setup.py sdist bdist_wheel'
  sh 'twine upload dist/*'
end
