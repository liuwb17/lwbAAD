"""DSH usage is reported, but exempt from the solver token budget."""


def budget_tokens(row):
    if row.get('budget_scope') == 'solver-plus-meta':
        return row.get('usage', {}).get('total_tokens', row.get('reserved_tokens', 0))
    if row.get('backend') == 'dsh':
        return 0
    return row.get('usage', {}).get('total_tokens', row.get('reserved_tokens', 0))


def dsh_totals(ledger):
    rows = [r for r in ledger if r.get('backend') == 'dsh']
    return {'sessions': len(rows),
            'reported_tokens': sum(r.get('dsh_usage', {}).get('reported_tokens', 0) for r in rows),
            'model_attempts': sum(r.get('dsh_usage', {}).get('model_attempts', 0) for r in rows),
            'incomplete_usage_sessions': sum(not r.get('dsh_usage', {}).get('complete', False) for r in rows),
            'budget_tokens': sum(budget_tokens(r) for r in rows)}
