# SPDX-FileCopyrightText: 2026-present Nace AI
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from haystack_integrations.common.nace.concurrency import map_in_threads


def test_map_in_threads_stops_after_first_failure():
    started = []

    def work(item):
        started.append(item)
        if item == 0:
            msg = "boom"
            raise RuntimeError(msg)
        return item

    with pytest.raises(RuntimeError, match="boom"):
        map_in_threads(work, list(range(10)), max_workers=1)
    assert started == [0]


def test_map_in_threads_stops_when_a_later_item_fails_while_an_earlier_one_is_slow():
    # Executor.map only notices a failure once every earlier item is done, so it would start all 20 here
    started = []

    def work(item):
        started.append(item)
        if item == 0:
            time.sleep(0.5)
        if item == 1:
            msg = "boom"
            raise RuntimeError(msg)
        return item

    with pytest.raises(RuntimeError, match="boom"):
        map_in_threads(work, list(range(20)), max_workers=2)
    assert len(started) <= 3


def test_map_in_threads_keeps_order():
    assert map_in_threads(lambda item: item * 2, [3, 1, 2], max_workers=3) == [6, 2, 4]


def test_map_in_threads_with_no_items():
    assert map_in_threads(lambda item: item, [], max_workers=3) == []
