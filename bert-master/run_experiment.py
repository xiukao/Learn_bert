# coding=utf-8
"""Run one BERT paper task and record its settings and results."""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import experiment_utils
import task_metrics


TASKS = {
    "MNLI": ("run_classifier.py", "mnli"),
    "QQP": ("run_classifier.py", "qqp"),
    "QNLI": ("run_classifier.py", "qnli"),
    "SST-2": ("run_classifier.py", "sst-2"),
    "CoLA": ("run_classifier.py", "cola"),
    "STS-B": ("run_classifier.py", "sts-b"),
    "MRPC": ("run_classifier.py", "mrpc"),
    "RTE": ("run_classifier.py", "rte"),
    "SQuAD1.1": ("run_squad.py", None),
    "SQuAD2.0": ("run_squad.py", None),
    "SWAG": ("run_swag.py", None),
}


def _utc_now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _git(repo_root, *arguments):
  result = subprocess.run(
      ["git", "-C", repo_root] + list(arguments),
      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      universal_newlines=True)
  return result.stdout.strip() if result.returncode == 0 else None


def _version(package):
  try:
    from importlib import metadata
  except ImportError:
    try:
      import pkg_resources
      return pkg_resources.get_distribution(package).version
    except Exception:
      return None
  try:
    return metadata.version(package)
  except metadata.PackageNotFoundError:
    return None


def _file_hashes(project_dir):
  result = {}
  for name in sorted(os.listdir(project_dir)):
    if not name.endswith(".py"):
      continue
    path = os.path.join(project_dir, name)
    with open(path, "rb") as stream:
      result[name] = hashlib.sha256(stream.read()).hexdigest()
  return result


def _create_run_directory(output_root, task):
  task_dir = os.path.join(output_root, task)
  os.makedirs(task_dir, exist_ok=True)
  number = 1
  while True:
    run_dir = os.path.join(task_dir, "%s-%d" % (task, number))
    try:
      os.mkdir(run_dir)
      return run_dir
    except FileExistsError:
      number += 1


def _has_flag(arguments, name):
  return any(value == "--" + name or value.startswith("--" + name + "=")
             for value in arguments)


def _read_eval_results(path):
  result = {}
  if not os.path.isfile(path):
    return result
  with open(path, "r") as stream:
    for line in stream:
      if "=" not in line:
        continue
      name, value = line.split("=", 1)
      name = name.strip()
      value = value.strip()
      try:
        numeric = float(value)
        result[name] = (int(numeric) if numeric.is_integer() else numeric)
      except ValueError:
        result[name] = value
  return result


def _collect_metrics(task, run_dir):
  metrics = {}
  dev = _read_eval_results(os.path.join(run_dir, "eval_results.txt"))
  if dev:
    metrics["dev"] = dev
  mismatched = _read_eval_results(
      os.path.join(run_dir, "eval_mismatched_results.txt"))
  if mismatched:
    metrics["dev_mismatched"] = mismatched
  note = None
  if task.startswith("SQuAD"):
    config_path = os.path.join(run_dir, "config.json")
    prediction_path = os.path.join(run_dir, "predictions.json")
    if os.path.isfile(config_path) and os.path.isfile(prediction_path):
      with open(config_path, "r") as stream:
        config = json.load(stream)
      predict_file = config.get("effective_flags", {}).get("predict_file")
      if predict_file:
        if not os.path.isabs(predict_file):
          predict_file = os.path.join(os.path.dirname(__file__), predict_file)
        try:
          metrics["dev"] = task_metrics.score_squad(
              predict_file, prediction_path)
          if task == "SQuAD2.0":
            metrics["dev"].update(task_metrics.best_squad2_thresholds(
                predict_file,
                os.path.join(run_dir, "nbest_predictions.json"),
                os.path.join(run_dir, "null_odds.json")))
        except (IOError, ValueError, KeyError) as error:
          note = "SQuAD gold scoring unavailable: %s" % error
  return metrics, note


def main():
  parser = argparse.ArgumentParser(
      description="Run one of the BERT paper's 11 downstream tasks. "
                  "Other --flags are passed to the task script.")
  parser.add_argument("--task", required=True,
                      help="One of: %s" % ", ".join(TASKS))
  parser.add_argument("--output_root", default="output",
                      help="Root directory for numbered task runs.")
  args, task_flags = parser.parse_known_args()
  task = next((name for name in TASKS
               if name.lower() == args.task.lower()), None)
  if task is None:
    parser.error("Unknown task: %s" % args.task)
  for reserved in ("output_dir", "task_name", "version_2_with_negative"):
    if _has_flag(task_flags, reserved):
      parser.error("--%s is controlled by run_experiment.py" % reserved)

  project_dir = os.path.dirname(os.path.abspath(__file__))
  output_root = (args.output_root if os.path.isabs(args.output_root) else
                 os.path.join(project_dir, args.output_root))
  run_dir = _create_run_directory(output_root, task)
  script, classifier_name = TASKS[task]
  command = [sys.executable, os.path.join(project_dir, script)] + task_flags
  if classifier_name is not None:
    command.append("--task_name=%s" % classifier_name)
  if task.startswith("SQuAD"):
    command.append("--version_2_with_negative=%s" %
                   ("true" if task == "SQuAD2.0" else "false"))
  if not any(_has_flag(task_flags, flag)
             for flag in ("do_train", "do_eval", "do_predict")):
    command.append("--do_train=true")
    command.append("--do_predict=true" if task.startswith("SQuAD")
                   else "--do_eval=true")
  command.append("--output_dir=%s" % run_dir)

  experiment_utils.write_json(
      os.path.join(run_dir, "config.json"),
      {"task": task, "user_flags": task_flags,
       "output_dir": run_dir,
       "note": "effective_flags is added by the task script before execution"})
  repo_root = os.path.dirname(project_dir)
  run_info = {
      "task": task,
      "started_at_utc": _utc_now(),
      "command": command,
      "working_directory": project_dir,
      "python": sys.version,
      "platform": platform.platform(),
      "dependencies": {
          name: _version(name) for name in
          ("tensorflow", "numpy", "six", "tensorflow-hub")},
      "git": {
          "branch": _git(repo_root, "branch", "--show-current"),
          "head": _git(repo_root, "rev-parse", "HEAD"),
          "status_at_start": _git(repo_root, "status", "--short"),
      },
      "python_file_sha256": _file_hashes(project_dir),
      "status": "running",
  }
  run_info_path = os.path.join(run_dir, "run_info.json")
  experiment_utils.write_json(run_info_path, run_info)

  started = time.monotonic()
  return_code = None
  error = None
  try:
    with open(os.path.join(run_dir, "run.log"), "w",
              encoding="utf-8") as log:
      process = subprocess.Popen(
          command, cwd=project_dir, stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT, universal_newlines=True,
          errors="replace")
      try:
        for line in process.stdout:
          log.write(line)
          log.flush()
          try:
            sys.stdout.write(line)
            sys.stdout.flush()
          except UnicodeEncodeError:
            pass
        return_code = process.wait()
      except KeyboardInterrupt:
        process.terminate()
        return_code = process.wait()
        raise
  except BaseException as exception:
    error = "%s: %s" % (type(exception).__name__, exception)
    if return_code is None:
      return_code = 130 if isinstance(exception, KeyboardInterrupt) else 1
  metrics, metrics_note = _collect_metrics(task, run_dir)
  status = "completed" if return_code == 0 else "failed"
  with open(os.path.join(run_dir, "config.json"), "r") as stream:
    effective_config = json.load(stream).get("effective_flags", {})
  dataset_paths = {
      key: effective_config[key] for key in
      ("data_dir", "train_file", "predict_file")
      if effective_config.get(key)}
  artifacts = []
  for name in sorted(os.listdir(run_dir)):
    if os.path.isfile(os.path.join(run_dir, name)):
      artifacts.append(name)
  metrics_record = {
      "task": task, "status": status, "datasets": dataset_paths,
      "metrics": metrics, "artifacts": artifacts,
      "note": metrics_note or
              ("No evaluation metrics were produced" if not metrics else None),
  }
  if error:
    metrics_record["error"] = error
  elif return_code:
    metrics_record["error"] = (
        "Task script exited with code %d; see run.log" % return_code)
  experiment_utils.write_json(
      os.path.join(run_dir, "metrics.json"), metrics_record)
  run_info.update({
      "finished_at_utc": _utc_now(),
      "elapsed_seconds": round(time.monotonic() - started, 3),
      "exit_code": return_code,
      "status": status,
  })
  experiment_utils.write_json(run_info_path, run_info)
  print("Run directory: %s" % run_dir)
  return return_code


if __name__ == "__main__":
  sys.exit(main())
