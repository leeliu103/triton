"""Queue-contract-driven waits at actual register consumers.

The scheduler is independent of register numbers and instruction spelling.
Its caller supplies access decoding and explicit ordered-completion queues.
This is deliberately not a universal AMD memory model: unordered queues,
aliasing memory stores, and control-flow joins require separate contracts.
"""
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Request:
    name: str
    writes: frozenset[int]


@dataclass(frozen=True)
class Access:
    reads: frozenset[int] = frozenset()
    writes: frozenset[int] = frozenset()
    request_queue: str | None = None


@dataclass(frozen=True)
class QueueContract:
    wait_opcode: str
    initially_pending: tuple[Request, ...] = ()
    max_wait_threshold: int = 63
    ordered_completion: bool = True


def rewrite_consumer_waits(
    stream: Sequence[str],
    queues: Mapping[str, QueueContract],
    decode: Callable[[str], Access],
    flush_before: Mapping[int, Sequence[str]] | None = None,
):
    """Replace waits for managed queues with just-sufficient consumer waits.

    `stream` is one path with explicit entry queue state. A caller may include
    a conditional request block only when it has reconciled all managed
    dependencies at its join (for example, drain current A before next B).
    `flush_before` supplies required fences, including boundaries where
    several possible request histories must agree. Existing managed waits
    are removed; memory fences must therefore be outside this stream or
    explicitly provided. All other instructions preserve their order.

    Whole-register RAW and WAW hazards are protected. The threshold equals
    the number of requests newer than the youngest conflicting request.
    Retired requests are removed conservatively, assuming no spontaneous
    completion. No speculative latency estimate enters the algorithm.
    """
    assert queues and all(q.ordered_completion for q in queues.values())
    pending = {name: list(q.initially_pending) for name, q in queues.items()}
    waits = {q.wait_opcode: name for name, q in queues.items()}
    assert len(waits) == len(queues)
    result, log = [], []
    flush_before = flush_before or {}

    def wait(name, threshold, index, reason):
        threshold = min(threshold, queues[name].max_wait_threshold)
        assert threshold >= 0
        if len(pending[name]) <= threshold:
            return
        retired = pending[name][:len(pending[name])-threshold]
        result.append(f'{queues[name].wait_opcode} 0x{threshold:x}')
        log.append({'before_input_index': index, 'queue': name,
                    'threshold': threshold, 'reason': reason,
                    'retires': [r.name for r in retired]})
        del pending[name][:len(pending[name])-threshold]

    for index, op in enumerate(stream):
        for name in flush_before.get(index, ()):
            wait(name, 0, index, 'explicit queue fence')
        if op.split(' ', 1)[0] in waits:
            continue
        access = decode(op)
        touched = access.reads | access.writes
        for name, requests in pending.items():
            conflicts = [i for i, r in enumerate(requests) if r.writes & touched]
            if conflicts:
                wait(name, len(requests)-max(conflicts)-1, index, 'register RAW/WAW')
        result.append(op)
        if access.request_queue in queues:
            pending[access.request_queue].append(Request(
                f'{access.request_queue}@{index}', access.writes))
    for name in flush_before.get(len(stream), ()):
        wait(name, 0, len(stream), 'explicit queue exit fence')
    return result, {'waits': log, 'pending_at_exit': {
        name: [{'name': r.name, 'writes': sorted(r.writes)} for r in requests]
        for name, requests in pending.items()}}
