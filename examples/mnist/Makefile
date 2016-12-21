var_dir = var
data_dir = var/data
train_file = ${data_dir}/train.tfrecords
valid_file = ${data_dir}/validation.tfrecords

options = \
	--train_file ${train_file} \
	--eval_file ${valid_file} \
	--output_dir ${var_dir}/output \
	--ps_hosts localhost:4242 \
	--worker_hosts localhost:5151 \
	--task_index 0


.PHONY: main
main: ${train_file} ${valid_file}
	python3 mnist.py ${options} --task_type ps &
	python3 mnist.py ${options} --task_type worker &
	python3 mnist.py ${options} --task_type master

convert_to_records.py:
	wget https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/$@

${train_file} ${valid_file}: convert_to_records.py
	python3 $< --directory ${data_dir}

.PHONY: clean
clean:
	rm -rf var/* convert_to_records.py
