"""Task metrics used by the paper's GLUE and SQuAD experiments."""

from __future__ import division

import collections
import json
import math
import re
import string


def _pearson(gold, predicted):
  if len(gold) < 2:
    return 0.0
  gold_mean = sum(gold) / len(gold)
  pred_mean = sum(predicted) / len(predicted)
  numerator = sum((x - gold_mean) * (y - pred_mean)
                  for x, y in zip(gold, predicted))
  gold_norm = sum((x - gold_mean) ** 2 for x in gold)
  pred_norm = sum((y - pred_mean) ** 2 for y in predicted)
  denominator = math.sqrt(gold_norm * pred_norm)
  return numerator / denominator if denominator else 0.0


def _average_ranks(values):
  ranks = [0.0] * len(values)
  order = sorted(range(len(values)), key=lambda index: values[index])
  start = 0
  while start < len(order):
    end = start + 1
    while end < len(order) and values[order[end]] == values[order[start]]:
      end += 1
    rank = (start + end - 1) / 2.0 + 1.0
    for position in order[start:end]:
      ranks[position] = rank
    start = end
  return ranks


def glue_metrics(task_name, gold, predicted):
  if len(gold) != len(predicted) or not gold:
    raise ValueError("Metric inputs must have equal, non-zero length")
  task_name = task_name.lower()
  if task_name == "sts-b":
    gold = [float(value) for value in gold]
    predicted = [float(value) for value in predicted]
    pearson = _pearson(gold, predicted)
    spearman = _pearson(_average_ranks(gold), _average_ranks(predicted))
    return {"pearson": pearson, "spearman": spearman,
            "correlation_mean": (pearson + spearman) / 2.0,
            "count": len(gold)}

  accuracy = sum(x == y for x, y in zip(gold, predicted)) / len(gold)
  result = {"accuracy": accuracy, "count": len(gold)}
  if task_name in ("mrpc", "qqp"):
    true_positive = sum(x == 1 and y == 1 for x, y in zip(gold, predicted))
    false_positive = sum(x != 1 and y == 1 for x, y in zip(gold, predicted))
    false_negative = sum(x == 1 and y != 1 for x, y in zip(gold, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    result["f1"] = (2 * true_positive / denominator
                    if denominator else 0.0)
    result["accuracy_f1_mean"] = (accuracy + result["f1"]) / 2.0
  if task_name == "cola":
    true_positive = sum(x == 1 and y == 1 for x, y in zip(gold, predicted))
    true_negative = sum(x == 0 and y == 0 for x, y in zip(gold, predicted))
    false_positive = sum(x == 0 and y == 1 for x, y in zip(gold, predicted))
    false_negative = sum(x == 1 and y == 0 for x, y in zip(gold, predicted))
    denominator = math.sqrt(
        (true_positive + false_positive) *
        (true_positive + false_negative) *
        (true_negative + false_positive) *
        (true_negative + false_negative))
    result["matthews_correlation"] = (
        (true_positive * true_negative - false_positive * false_negative) /
        denominator if denominator else 0.0)
  return result


def _normalize_answer(answer):
  answer = answer.lower()
  answer = "".join(character for character in answer
                   if character not in string.punctuation)
  answer = re.sub(r"\b(a|an|the)\b", " ", answer)
  return " ".join(answer.split())


def _squad_pair_scores(prediction, reference):
  prediction = _normalize_answer(prediction)
  reference = _normalize_answer(reference)
  exact_match = float(prediction == reference)
  prediction_tokens = prediction.split()
  reference_tokens = reference.split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      reference_tokens)
  same_tokens = sum(common.values())
  if not prediction_tokens or not reference_tokens:
    return exact_match, float(prediction_tokens == reference_tokens)
  if not same_tokens:
    return exact_match, 0.0
  precision = same_tokens / len(prediction_tokens)
  recall = same_tokens / len(reference_tokens)
  return exact_match, 2 * precision * recall / (precision + recall)


def score_squad(gold_file, prediction_file):
  """Score dev predictions with SQuAD's normalized EM/F1 definitions."""
  with open(gold_file, encoding="utf-8") as stream:
    dataset = json.load(stream)
  with open(prediction_file, encoding="utf-8") as stream:
    predictions = json.load(stream)
  exact_sum = 0.0
  f1_sum = 0.0
  count = 0
  missing = 0
  for article in dataset["data"]:
    for paragraph in article["paragraphs"]:
      for question in paragraph["qas"]:
        question_id = question["id"]
        if "answers" not in question and not question.get("is_impossible"):
          raise ValueError("SQuAD file has no gold answers for %s" %
                           question_id)
        answers = [item["text"] for item in question.get("answers", [])]
        if not answers:
          answers = [""]
        if question_id not in predictions:
          missing += 1
        prediction = predictions.get(question_id, "")
        scores = [_squad_pair_scores(prediction, answer)
                  for answer in answers]
        exact_sum += max(score[0] for score in scores)
        f1_sum += max(score[1] for score in scores)
        count += 1
  if not count:
    raise ValueError("SQuAD file contains no questions")
  return {"exact_match": 100.0 * exact_sum / count,
          "f1": 100.0 * f1_sum / count,
          "count": count,
          "missing_predictions": missing}


def best_squad2_thresholds(gold_file, nbest_file, null_odds_file):
  """Select EM and F1 no-answer thresholds on a labeled dev split."""
  with open(gold_file, encoding="utf-8") as stream:
    dataset = json.load(stream)
  with open(nbest_file, encoding="utf-8") as stream:
    nbest = json.load(stream)
  with open(null_odds_file, encoding="utf-8") as stream:
    null_odds = json.load(stream)
  rows = []
  for article in dataset["data"]:
    for paragraph in article["paragraphs"]:
      for question in paragraph["qas"]:
        question_id = question["id"]
        if "answers" not in question and not question.get("is_impossible"):
          raise ValueError("SQuAD file has no gold answers for %s" %
                           question_id)
        answers = [item["text"] for item in question.get("answers", [])] or [""]
        candidates = nbest.get(question_id, [])
        non_null = next((item["text"] for item in candidates
                         if item["text"]), "")
        if question_id not in null_odds:
          raise ValueError("Missing null score for %s" % question_id)
        null_scores = [_squad_pair_scores("", answer) for answer in answers]
        answer_scores = [_squad_pair_scores(non_null, answer)
                         for answer in answers]
        rows.append((
            float(null_odds[question_id]),
            max(item[0] for item in null_scores),
            max(item[1] for item in null_scores),
            max(item[0] for item in answer_scores),
            max(item[1] for item in answer_scores)))
  if not rows:
    raise ValueError("SQuAD file contains no questions")
  rows.sort(key=lambda item: item[0])
  exact_total = sum(item[1] for item in rows)
  f1_total = sum(item[2] for item in rows)
  best_exact = (exact_total, rows[0][0] - 1.0)
  best_f1 = (f1_total, rows[0][0] - 1.0)
  position = 0
  while position < len(rows):
    threshold = rows[position][0]
    while position < len(rows) and rows[position][0] == threshold:
      _, null_exact, null_f1, answer_exact, answer_f1 = rows[position]
      exact_total += answer_exact - null_exact
      f1_total += answer_f1 - null_f1
      position += 1
    if exact_total > best_exact[0]:
      best_exact = (exact_total, threshold)
    if f1_total > best_f1[0]:
      best_f1 = (f1_total, threshold)
  return {
      "best_exact_match": 100.0 * best_exact[0] / len(rows),
      "best_exact_threshold": best_exact[1],
      "best_f1": 100.0 * best_f1[0] / len(rows),
      "best_f1_threshold": best_f1[1],
  }
