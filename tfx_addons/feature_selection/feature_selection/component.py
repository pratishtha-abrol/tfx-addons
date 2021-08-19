# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib
import tensorflow as tf
import os
from tfx.dsl.component.experimental.decorators import component
from tfx.types import artifact, artifact_utils
from tfx.types.standard_artifacts import Examples
from tfx.v1.dsl.components import OutputArtifact, InputArtifact, Parameter
# TODO: Why does import from tfx.dsl.components not work?


"""Custom Artifact type"""
class FeatureSelectionArtifact(artifact.Artifact):
  """Output artifact containing feature scores from the Feature Selection component"""
  TYPE_NAME = 'Feature Selection'
  PROPERTIES = {
      'scores': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'p_values': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'selected_features': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'selected_data': artifact.Property(type=artifact.PropertyType.JSON_VALUE)
  }


# TODO: Remove the need to declare extract_fn -- use tfx_bsl example_coder
def extract_fn(data_record):
    features = {
        'species': tf.io.FixedLenFeature([], tf.int64),
        'culmen_length_mm':tf.io.FixedLenFeature([], tf.float32),
        'culmen_depth_mm':tf.io.FixedLenFeature([], tf.float32),
        'flipper_length_mm':tf.io.FixedLenFeature([], tf.float32),
        'body_mass_g':tf.io.FixedLenFeature([], tf.float32)
    }
    sample = tf.io.parse_single_example(data_record, features) 
    return sample


def get_data_from_TFRecords(train_uri, target_feature):
    train_uri = [os.path.join(train_uri, 'data_tfrecord-00000-of-00001.gz')]
    raw_dataset = tf.data.TFRecordDataset(train_uri, compression_type='GZIP')
    dataset = raw_dataset.map(extract_fn)
    np_dataset = list(dataset.as_numpy_iterator())

    feature_keys = list(np_dataset[0].keys())
    target = [i.pop(target_feature) for i in np_dataset]
    input_data = [list(i.values()) for i in np_dataset]

    return [feature_keys, target, input_data]


"""
Feature selection component
"""
@component
def FeatureSelection(module_file: Parameter[str],
    orig_examples: InputArtifact[Examples],
    feature_selection: OutputArtifact[FeatureSelectionArtifact]):
  """Feature Selection component
      Args (from the module file):
        NUM_PARAM: Parameter for the corresponding mode in SelectorFunc
          example: value of 'k' in SelectKBest
        TARGET_FEATURE: Name of the feature containing target data
        SelectorFunc: Selector function for univariate feature selection
          example: SelectKBest, SelectPercentile from sklearn.feature_selection
        ScoreFunc: Scoring function for various features with INPUT_DATA and OUTPUT_DATA as parameters
  """

  train_uri = artifact_utils.get_split_uri([orig_examples], 'train')

  # importing the required functions and variables from
  modules = importlib.import_module(module_file)
  mod_names = ["NUM_PARAM", "TARGET_FEATURE", "SelectorFunc", "ScoreFunc"]
  NUM_PARAM, TARGET_FEATURE, SelectorFunc, ScoreFunc = [getattr(modules, i) for i in mod_names]

  FEATURE_KEYS, TARGET_DATA, INPUT_DATA = get_data_from_TFRecords(train_uri, TARGET_FEATURE)

  # Select features based on scores
  selector = SelectorFunc(ScoreFunc, k=NUM_PARAM)
  selected_data = selector.fit_transform(INPUT_DATA, TARGET_DATA).tolist()

  # generate a list of selected features by matching _FEATURE_KEYS to selected indices
  selected_features = [val for (idx, val) in enumerate(FEATURE_KEYS) if idx in selector.get_support(indices=True)]

  # get scores and p-values for artifacts
  selector_scores = selector.scores_
  selector_p_values = selector.pvalues_

  # merge scores and pvalues with feature keys to create a dictionary
  selector_scores_dict = dict(zip(FEATURE_KEYS, selector_scores))
  selector_pvalues_dict = dict(zip(FEATURE_KEYS, selector_p_values))

  # populate artifact with the required properties
  feature_selection.scores = selector_scores_dict
  feature_selection.p_values = selector_pvalues_dict
  feature_selection.selected_features = selected_features
  feature_selection.selected_data = selected_data
