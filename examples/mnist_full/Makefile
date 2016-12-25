var_dir = var
data_dir = ${var_dir}/data
train_file = ${data_dir}/train.tfrecords
valid_file = ${data_dir}/validation.tfrecords
script = mnist_full.py

mnist = python3 ${script} \
	--train_steps 1000 \
	--train_file ${train_file} \
	--eval_file ${valid_file} \
	--output_dir ${var_dir}/output \
	--master_host localhost:2049 \
	--ps_hosts localhost:4242 \
	--task_type
kill_servers = kill $$(ps x | grep ${script} | grep -v grep | awk '{print $$1}') || :


.PHONY: main
main: ${train_file} ${valid_file}
	${kill_servers}
	${mnist} ps > ${var_dir}/ps.log 2>&1 &
	${mnist} master
	${kill_servers}


${train_file} ${valid_file}:
	curl -SL https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/convert_to_records.py | \
	python3 - --directory ${data_dir}


.PHONY: clean
clean:
	git clean -dfx .
