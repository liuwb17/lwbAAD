"""Bounded score feedback for prompts; raw measurements remain in saved state."""
from copy import deepcopy

FEEDBACK_PROTOCOL = 'scores-timing-bounded-errors-no-raw-io-1'
ERROR_CHARS = 2000


def score_row(row, *, diagnostic=False):
    result = {key: row[key] for key in ('name', 'score', 'raw_score', 'sample_raw_score',
                                       'seconds', 'seed', 'status') if key in row}
    if diagnostic and row['status'] != 'AC':
        message = row.get('message', '') or ''
        result['message'] = (message if len(message) <= ERROR_CHARS else
                             message[:1000] + '\n...[truncated]...\n' + message[-980:])
    return result


def score_result(result):
    if result is None:
        return None
    summary = {key: result[key] for key in ('valid', 'raw_mean', 'mean', 'evaluation_count', 'seed')
               if key in result}
    diagnostics = 0
    rows = []
    for row in result.get('cases', []):
        include = row['status'] != 'AC' and diagnostics < 2
        rows.append(score_row(row, diagnostic=include))
        diagnostics += include
    summary['cases'] = rows
    return summary


def score_record(record):
    return {key: (score_result(value) if key in ('full', 'smoke') else
                  [score_result(item) for item in value] if key == 'evaluations' else deepcopy(value))
            for key, value in record.items()}
