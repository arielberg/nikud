import tensorflow as tf
import numpy as np
import apache_beam as beam
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
import utils
import os
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

np.set_printoptions(linewidth=460)

tf.logging.set_verbosity(tf.logging.INFO)
def train_input_fn(dataset):
    """
    chunk = chunks.popleft()
    labels = chunk.pop('labels')
    """
    ds = tf.data.Dataset.from_tensor_slices((dict(dataset['word_is_beginig']), dataset['labels']))


    return ds.make_one_shot_iterator().get_next()


def prepareCSV(*args):
    return (dict(args), args[3])

def preprocessing_fn(inputs):
    outputs = inputs.copy()
    labels = outputs['labels']
    del outputs['labels']
    return labels, outputs

def _make_training_input_fn(tf_transform_output, transformed_examples,
                            batch_size):
    """Creates an input function reading from transformed data.

    Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    transformed_examples: Base filename of examples.
    batch_size: Batch size.

    Returns:
    The input function for training or eval.
    """
    def input_fn():
        """Input function for training and eval."""
        dataset = tf.contrib.data.make_batched_features_dataset(
            file_pattern=transformed_examples,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            shuffle=True)

        transformed_features = dataset.make_one_shot_iterator().get_next()

        # Extract features and label from the transformed tensors.
        transformed_labels = transformed_features.pop(LABEL_KEY)

        return transformed_features, transformed_labels

    return input_fn


def do_train():
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir='tmp'):
            RAW_DATA_FEATURE_SPEC = dict(
                [(name, tf.FixedLenFeature([], tf.float32))
                 for name in CSV_COLUMNS_TRAIN]
            )

            RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
                dataset_schema.from_feature_spec(RAW_DATA_FEATURE_SPEC)
            )
            converter = tft.coders.CsvCoder(CSV_COLUMNS_TRAIN, RAW_DATA_METADATA.schema)


            raw_data = (
                    pipeline
                    | 'ReadTrainData' >> beam.io.ReadFromText(utils.vector_CSV_file)
                    | 'FixCommasTrainData' >> beam.Map(
                lambda line: line.replace(', ', ','))
                    | 'DecodeTrainData' >> MapAndFilterErrors(converter.decode))

            my_feature_columns = []
            for key in CSV_COLUMNS_TRAIN:
                if key != "labels" and "ignore" not in key:
                    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

            classifier = tf.estimator.DNNClassifier(
                feature_columns=my_feature_columns,
                # Three hidden layers of 10 nodes.
                hidden_units=[50, 50],
                # The model must choose between 2 classes.
                n_classes=len(utils.nikudStr) + 1)

            tf_transform_output = tft.TFTransformOutput('tmp')

            train_input_fn = _make_training_input_fn(
                tf_transform_output,
                os.path.join('tmp', 'train_transformed' + '*'),
                batch_size=1)
            classifier.train(input_fn=train_input_fn)

    my_feature_columns = []
    for key in CSV_COLUMNS_TRAIN:
        if key != "labels" and "ignore" not in key:
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    dataset = tf.data.experimental.CsvDataset(utils.vector_CSV_file,
                                        [tf.float32  for i in range(len(my_feature_columns))],
                                        header=True, buffer_size=1)

    with tft_beam.Context(temp_dir='tmp'):
        transformed_dataset, transform_fn = (
                dataset | tft_beam.AnalyzeAndTransformDataset(
                    preprocessing_fn))

    # dataset.map(prepareCSV)
    dataset.shuffle(1000)

    """
    df = pd.read_csv(utils.vector_CSV_file, names=CSV_COLUMNS_TRAIN, header=None, chunksize=1000)
    from collections import deque
    df_que = deque(df)

    """
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Three hidden layers of 10 nodes.
        hidden_units=[50, 50],
        # The model must choose between 2 classes.
        n_classes=len(utils.nikudStr)+1)

    classifier.train(input_fn=lambda: train_input_fn(dataset))

    model.save_weights("weights/model_100X50_B_"+str(epochId)+".h5")


CSV_COLUMNS_TRAIN = ['word_is_beginig',
                     'word_is_ending',
                     'char_place_in_word',
                     'labels']

""" TODO : Remove ID Coulmns """
for i in range(len(utils.nikudStr)):
    CSV_COLUMNS_TRAIN.append("ignore_"+str(i))

for i in range(len(utils.chars)):
    CSV_COLUMNS_TRAIN.append("char_is_"+str(i))
for n in range(utils.charBefore):
    for i in range(len(utils.chars)):
        CSV_COLUMNS_TRAIN.append("before_"+str(n+1)+"_is_"+str(i))
for n in range(utils.charAfter):
    for i in range(len(utils.chars)):
        CSV_COLUMNS_TRAIN.append("after_"+str(n+1)+"_is_"+str(i))
for i in range(100):
    CSV_COLUMNS_TRAIN.append("word_to_vec_"+str(i))

chunkID = -1
chunkedDS = None

class MapAndFilterErrors(beam.PTransform):
    """Like beam.Map but filters out erros in the map_fn."""

    class _MapAndFilterErrorsDoFn(beam.DoFn):
        """Count the bad examples using a beam metric."""

        def __init__(self, fn):
            self._fn = fn
            # Create a counter to measure number of bad elements.
            self._bad_elements_counter = beam.metrics.Metrics.counter('rel_example', 'bad_elements')

    def process(self, element):
        try:
            yield self._fn(element)
        except Exception:  # pylint: disable=broad-except
            # Catch any exception the above call.
            self._bad_elements_counter.inc(1)

    def __init__(self, fn):
        self._fn = fn

    def expand(self, pcoll):
        return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))

if __name__== "__main__":
    do_train()

