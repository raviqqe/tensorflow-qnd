task :module_test do
  `find qnd`.split.each do |test_mod|
    if test_mod =~ /_test\.py/
      mod = test_mod.gsub('/', '.').gsub('_test.py', '')

      puts "Testing #{mod}"
      sh "python3 -m #{mod}_test"
    end
  end
end


task :mnist_example => :clean do
  venv_dir = 'venv'

  sh "python3 -m venv #{venv_dir}"

  [
    'pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp35-cp35m-linux_x86_64.whl',
    'python3 setup.py install',
    'make -C examples/mnist',
  ].each do |command|
    sh ". #{venv_dir}/bin/activate && #{command}"
  end
end


task :test => %i(module_test mnist_example)


task :upload => %i(test clean) do
  sh 'python3 setup.py sdist bdist_wheel'
  sh 'twine upload dist/*'
end


task :clean do
  sh 'git clean -dfx'
end
