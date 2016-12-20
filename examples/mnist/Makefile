options = \
	--train_file /tmp/data/train.tfrecords \
	--eval_file /tmp/data/validation.tfrecords \
	--ps_hosts localhost:4242 \
	--worker_hosts localhost:5151 \
	--task_index 0


.PHONY: main
main: /tmp/data/train.tfrecords /tmp/data/validation.tfrecords
	python3 mnist.py ${options} --task_type ps &
	python3 mnist.py ${options} --task_type worker

convert_to_records.py:
	wget https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/$@

/tmp/data/train.tfrecords /tmp/data/validation.tfrecords: convert_to_records.py
	python3 $<

.PHONY: clean
clean:
	rm -f /tmp/data/*.tfrecords convert_to_records.py
