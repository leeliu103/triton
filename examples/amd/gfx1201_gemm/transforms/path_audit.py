"""Conservative global-register audit for concrete uniform loop paths."""
from collections import Counter
import re

from prepare_ready_waits import vgprs


def operations(text):
    result = []
    for raw in text.splitlines():
        op = raw.split(';', 1)[0].strip()
        if op and not op.startswith(('.loc', '.Ltmp')):
            result.append(op)
    return result


def uniform_path(stream, condition, loop_label='.LBB0_1'):
    """Resolve scc branches using the known current loop continuation bit."""
    labels = {op[:-1]: i for i, op in enumerate(stream) if op.endswith(':')}
    selected = []
    ip = 0
    while ip < len(stream):
        op = stream[ip]
        ip += 1
        if op.endswith(':'):
            continue
        if op.startswith(('s_cbranch_scc', 's_branch ')):
            opcode, target = op.split()
            if target == loop_label:
                break
            taken = opcode == 's_branch' or condition == bool(int(opcode[-1]))
            if taken:
                ip = labels[target]+1
            continue
        selected.append(op)
    return selected


def audit_global_paths(text, iterations=128):
    prologue, hot = text.split('.LBB0_1:', 1)
    hot = hot.split('s_cbranch_scc1 .LBB0_1', 1)[0]
    pending, request_histogram = [], Counter()
    checked = requests = 0

    def step(op):
        nonlocal checked, requests
        active = frozenset().union(*(p[1] for p in pending))
        if op.startswith('buffer_load_b128 '):
            dst, rest = op.split(',', 1)
            assert not vgprs(rest) & active, (op, 'RAW pending global')
            assert not vgprs(dst) & active, (op, 'WAW pending global')
            pending.append((requests, vgprs(dst)))
            requests += 1
            request_histogram[op] += 1
        elif op.startswith('s_wait_loadcnt '):
            threshold = int(op.split()[1], 0)
            del pending[:max(0, len(pending)-threshold)]
        elif op.startswith(('v_', 'ds_')):
            assert not vgprs(op) & active, (op, sorted(vgprs(op) & active))
            checked += 1

    for op in operations(prologue):
        if op.startswith('buffer_load_b128 '):
            step(op)
    states = []
    for iteration in range(iterations):
        for op in uniform_path(operations(hot), iteration < iterations-1):
            step(op)
        if iteration in (0, 1, iterations-1):
            states.append({'iteration': iteration, 'pending_at_exit': len(pending)})
    assert requests == iterations*12 and not pending, (requests, pending)
    assert set(request_histogram.values()) == {iterations}, request_histogram
    return {'passed': True, 'modeled_iterations': iterations,
            'global_requests': requests, 'unique_global_instructions': len(request_histogram),
            'each_global_request_executes_exactly_K_iterations': True,
            'checked_vector_LDS_instructions': checked, 'states': states,
            'assumption': 'ordered same-kind buffer_load_b128 completion count'}


def dynamic_vector_histogram(text, iterations=128):
    prologue, tail = text.split('.LBB0_1:', 1)
    hot, epilogue = tail.split('s_cbranch_scc1 .LBB0_1', 1)
    result = Counter(op for op in operations(prologue+epilogue) if op.startswith(('v_', 'ds_', 'buffer_')))
    for condition, count in ((True, iterations-1), (False, 1)):
        for op in uniform_path(operations(hot), condition):
            if op.startswith(('v_', 'ds_', 'buffer_')):
                result[op] += count
    return result
