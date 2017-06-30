import http.server
import json
import logging
import queue
import threading

import numpy as np
import tensorflow as tf

from .estimator import def_estimator
from .flag import FLAGS, add_flag, add_output_dir_flag


def def_serve():
    """Define `serve()` function.

    See also `help(def_serve())`.

    - Returns
        - `serve()` function.
    """
    add_output_dir_flag()
    add_flag('ip_address', default='')
    add_flag('port', type=int, default=80)

    estimator = def_estimator(distributed=False)

    def serve(model_fn, preprocess_fn=None, postprocess_fn=None):
        """Serve as a HTTP server.

        - Args
            - `model_fn`: Same as `train_and_evaluate()`'s.
            - `preprocess_fn`: A function to preprocess server request bodies
                in JSON. Its first argument is a function which returns the
                JSON input.
            - `preprocess_fn`: A function to postprocess server responses of
                JSON serializable objects.
        """
        server = EstimatorServer(
            estimator(model_fn, FLAGS.output_dir),
            preprocess_fn,
            postprocess_fn)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                inputs = json.loads(self.rfile.read(
                    int(self.headers['Content-Length'])))

                outputs = server.predict(inputs)

                logging.info('Prediction results: {}'.format(outputs))

                self.wfile.write(json.dumps(outputs).encode())

        http.server.HTTPServer((FLAGS.ip_address, FLAGS.port), Handler) \
            .serve_forever()

    return serve


def _make_json_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, dict):
        return {key: _make_json_serializable(value)
                for key, value in x.items()}
    elif isinstance(x, list):
        return [_make_json_serializable(value) for value in x]

    return x


class EstimatorServer:
    def __init__(self, estimator, preprocess_fn=None, postprocess_fn=None):
        self._input_queue = queue.Queue()
        self._output_queue = queue.Queue()

        def input_fn():
            return (tf.train.batch(preprocess_fn(self._input_queue.get), 1),
                    None)

        def target():
            for output in estimator.predict(input_fn=input_fn):
                self._output_queue.put(postprocess_fn(output))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def predict(self, inputs):
        self._input_queue.put(inputs)
        return self._output_queue.get()
