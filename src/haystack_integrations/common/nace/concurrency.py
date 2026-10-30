# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence
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


async def map_async(func: Callable[[T], Awaitable[R]], items: Sequence[T], max_workers: int) -> list[R]:
    """
    Applies an async function with bounded workers, stopping dispatch when a worker fails.

    :param func: The operation for one item.
    :param items: Inputs in result order.
    :param max_workers: Maximum concurrent operations.
    :returns: Results in input order.
    """
    queue = iter(enumerate(items))
    results: dict[int, R] = {}
    failed = False

    async def worker() -> None:
        nonlocal failed
        while not failed:
            entry = next(queue, None)
            if entry is None:
                return
            index, item = entry
            try:
                results[index] = await func(item)
            except BaseException:
                # Set before yielding so no other worker can take another queued item after this failure.
                failed = True
                raise

    tasks = [asyncio.create_task(worker()) for _ in range(min(max_workers, len(items)))]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [results[index] for index in range(len(items))]
