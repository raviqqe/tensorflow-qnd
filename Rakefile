VENV_DIR = '.venv'
TENSORFLOW_URL = 'https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp35-cp35m-linux_x86_64.whl'


def venv_sh *args, **kwargs
  sh([". #{VENV_DIR}/bin/activate &&", *args.map{ |x| x.to_s }].join(' '),
     **kwargs)
end


def pip_install *packages
  venv_sh 'pip3 install --upgrade', *packages
end


task :venv do
  sh "python3 -m venv #{VENV_DIR}" unless File.directory? VENV_DIR
end


task :clean do
  sh 'git clean -dfx'
end


task :new_venv => %i(clean venv)


task :module_test => :new_venv do
  pip_install 'gargparse', TENSORFLOW_URL
  `find .`.split.select do |file|
    file =~ /_test\.py$/ and file !~ /\/\./
  end.each do |file|
    sh 'pytest', file
  end
end


task :mnist_example => :new_venv do
  pip_install TENSORFLOW_URL

  ['python3 setup.py install', 'make -C examples/mnist'].each do |command|
    venv_sh command
  end
end


task :test => %i(module_test mnist_example)


task :upload => %i(test clean) do
  sh 'python3 setup.py sdist bdist_wheel'
  sh 'twine upload dist/*'
end
