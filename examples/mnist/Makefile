var_dir = var
data_dir = var/data
train_file = ${data_dir}/train.tfrecords
valid_file = ${data_dir}/validation.tfrecords

options = \
	--num_epochs 10 \
	--train_file ${train_file} \
	--eval_file ${valid_file} \
	--output_dir ${var_dir}/output \
	--master_host localhost:2049 \
	--ps_hosts localhost:4242 \
	--worker_hosts localhost:2049 \
	--task_index 0


.PHONY: main
main: ${train_file} ${valid_file}
	python3 mnist.py ${options} --task_type ps > ${var_dir}/ps.log 2>&1 &
	python3 mnist.py ${options} --task_type master 2>&1 | tee ${var_dir}/master.log

convert_to_records.py:
	wget https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/$@

${train_file} ${valid_file}: convert_to_records.py
	python3 $< --directory ${data_dir}

.PHONY: clean
clean:
	rm -rf var/* convert_to_records.py
