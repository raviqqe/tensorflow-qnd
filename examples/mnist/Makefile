var_dir = var
data_dir = ${var_dir}/data
train_file = ${data_dir}/train.tfrecords
valid_file = ${data_dir}/validation.tfrecords

mnist = python3 mnist.py \
	--num_epochs 10 \
	--train_file ${train_file} \
	--eval_file ${valid_file} \
	--output_dir ${var_dir}/output \
	--master_host localhost:2049 \
	--ps_hosts localhost:4242 \
	--task_type


.PHONY: main
main: ${train_file} ${valid_file}
	kill $$(ps x | grep mnist.py | grep -v grep | awk '{print $$1}') || :
	${mnist} ps > ${var_dir}/ps.log 2>&1 &
	${mnist} master 2>&1 | tee ${var_dir}/master.log


${train_file} ${valid_file}:
	curl -SL https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/convert_to_records.py | \
	python3 - --directory ${data_dir}

.PHONY: clean
clean:
	git clean -dfx .
