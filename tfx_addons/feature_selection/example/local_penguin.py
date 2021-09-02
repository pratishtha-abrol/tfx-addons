import os
import tempfile
import urllib
from typing import Text

import absl
import tfx
from tfx.components import CsvExampleGen, SchemaGen, StatisticsGen
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local import local_dag_runner

from tfx_addons.feature_selection.feature_selection.component import FeatureSelection

# downloading data and setting up required paths
_data_root = tempfile.mkdtemp(prefix='tfx-data')
DATA_PATH = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'
_data_filepath = os.path.join(_data_root, "data.csv")
urllib.request.urlretrieve(DATA_PATH, _data_filepath)

_pipeline_name = 'penguin_pipeline'
_tfx_root = tfx.__path__[0]
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     metadata_path: Text) -> pipeline.Pipeline:

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)

  # give path to the module file
  feature_selector = FeatureSelection(orig_examples = example_gen.outputs['examples'],
                                     module_file="module_file1")

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[example_gen, feature_selector],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path))


if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  local_dag_runner.LocalDagRunner().run(
      _create_pipeline(pipeline_name=_pipeline_name,
                       pipeline_root=_pipeline_root,
                       data_root=_data_root,
                       metadata_path=_metadata_path))
