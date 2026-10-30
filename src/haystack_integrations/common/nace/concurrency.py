# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_in_threads(func: Callable[[T], R], items: Sequence[T], max_workers: int) -> list[R]:
    """
    Applies `func` to every item on up to `max_workers` threads and returns the results in input order.

    Unlike `Executor.map`, which notices a failure only once every earlier item is done, the first exception from
    any completed item stops refilling the workers. Requests already in flight may still finish.

    :param func: The function to apply.
    :param items: The inputs.
    :param max_workers: The most items processed at once.
    :returns: `func(item)` for each item, in the order of `items`.
    :raises Exception: The first exception `func` raised, after the items already running have finished.
    """
    results: dict[int, R] = {}
    pending: dict[Future[R], int] = {}
    queue: Iterator[tuple[int, T]] = iter(enumerate(items))
    # Keep the queue here, rather than handing every item to the executor in advance.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        def start_next() -> None:
            for index, item in queue:
                pending[executor.submit(func, item)] = index
                return

        for _ in range(max_workers):
            start_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                error = future.exception()
                if error is not None:
                    raise error
            for future in done:
                results[pending.pop(future)] = future.result()
            for _ in done:
                start_next()
    return [results[index] for index in range(len(items))]
