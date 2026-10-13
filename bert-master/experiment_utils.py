"""Small, dependency-free helpers for recording effective experiment settings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os


def write_json(path, value):
  """Replace one JSON document atomically within its run directory."""
  directory = os.path.dirname(os.path.abspath(path))
  if not os.path.isdir(directory):
    os.makedirs(directory)
  temporary_path = path + ".tmp-%d" % os.getpid()
  with open(temporary_path, "w") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
  os.replace(temporary_path, path)


def _json_value(value):
  if isinstance(value, bytes):
    return value.decode("utf-8", "replace")
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  if isinstance(value, dict):
    return {str(key): _json_value(item) for key, item in value.items()}
  return str(value)


def save_effective_flags(flag_values, output_dir):
  """Save parsed TensorFlow flags, including defaults, before model execution."""
  if hasattr(flag_values, "flag_values_dict"):
    values = flag_values.flag_values_dict()
  else:
    values = {name: getattr(flag_values, name) for name in flag_values}
  path = os.path.join(output_dir, "config.json")
  document = {}
  if os.path.isfile(path):
    with open(path, "r") as stream:
      document = json.load(stream)
  document["effective_flags"] = _json_value(values)
  write_json(path, document)


def save_computed(output_dir, values):
  """Merge dataset-dependent training values into config.json."""
  path = os.path.join(output_dir, "config.json")
  with open(path, "r") as stream:
    document = json.load(stream)
  document.setdefault("computed", {}).update(_json_value(values))
  write_json(path, document)
