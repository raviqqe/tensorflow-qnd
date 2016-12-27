var_dir = var
data_dir = ${var_dir}/data
train_file = ${data_dir}/train.tfrecords
valid_file = ${data_dir}/validation.tfrecords


.PHONY: main
main: ${train_file} ${valid_file}
	${kill_servers}
	python3 mnist_full.py \
		--train_steps 1000 \
		--train_file ${train_file} \
		--eval_file ${valid_file} \
		--output_dir ${var_dir}/output
	${kill_servers}


${train_file} ${valid_file}:
	../lib/fetch_dataset.sh


.PHONY: clean
clean:
	git clean -dfx .
