# SPDX-FileCopyrightText: 2026-present Inria
# SPDX-License-Identifier: BSD-3-Clause

"""Over-the-air LH2 capture collection (swarmit transport).

A DotBot's secure bootloader samples its own raw LH2 counts on request
(READY mode only) and ships them back inside a SWARMIT_EVENT_LOG. This
module triggers the captures and decodes the records; the solve and the file
live in `lighthouse2`.

A capture is n reads per point, not one: the per-point error is the
placement sigma and the lighthouse's own over sqrt(n) in quadrature, so a
single read costs about 60 % at every point of the field. Every record of
every visible station in every reply is kept, because a point seen by two
stations is the co-visibility evidence a seam needs.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from dotbot.calibration.lighthouse2 import LH2CalibrationSample, Sample
from dotbot.calibration.points import CORNERS  # noqa: F401 - the capture order

CAPTURE_TIMEOUT_DEFAULT = 5.0
CAPTURE_RETRIES_DEFAULT = 3
CAPTURE_READS_DEFAULT = 25

# Each raw sample inside the LOG payload is [lh_index:1][count1:4 LE][count2:4 LE].
_SAMPLE_SIZE = 9


def parse_capture_payload(data: bytes, tag: int) -> list[LH2CalibrationSample]:
    """Decode a SWARMIT_EVENT_LOG payload of raw LH2 samples.

    Layout (mirrors the swarmit bootloader): a 1-byte `tag`, then N
    fixed-size records. Returns [] for any payload that is not a capture
    (regular text log lines do not carry `tag` as their first byte).
    """
    if len(data) < 1 or data[0] != tag:
        return []
    body = data[1:]
    samples: list[LH2CalibrationSample] = []
    for off in range(0, len(body) - _SAMPLE_SIZE + 1, _SAMPLE_SIZE):
        lh_index = body[off]
        count1 = int.from_bytes(body[off + 1 : off + 5], "little")
        count2 = int.from_bytes(body[off + 5 : off + 9], "little")
        samples.append(LH2CalibrationSample(lh_index, count1, count2))
    return samples


def samples_from_reads(
    reads: list[list[LH2CalibrationSample]], point: int
) -> list[Sample]:
    """Group per-capture records into one `Sample` per station at `point`.

    A station that a capture missed contributes fewer reads than the others,
    which the stillness guard and the solve both tolerate.
    """
    per_station: dict[int, Sample] = {}
    for capture in reads:
        for record in capture:
            sample = per_station.setdefault(
                record.lh_index, Sample(station=record.lh_index, point=point)
            )
            sample.count1.append(record.count1)
            sample.count2.append(record.count2)
    return [per_station[index] for index in sorted(per_station)]


class CaptureSession:
    """One shared log-event stream for a whole collect session.

    The bot only emits raw counts in reply to a trigger, so nothing arrives
    unsolicited - a single `watch_log_events()` stream serves every point. A
    background reader thread decodes records addressed to `device` into a
    queue; `capture()` triggers and waits, re-triggering on timeout because
    the trigger send is best-effort (no transport-level ack).
    """

    def __init__(self, client, device: str, tag: int):
        self._client = client
        self._device = device.upper()
        self._tag = tag
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def __enter__(self) -> CaptureSession:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()

    def _reader(self) -> None:
        try:
            for event in self._client.watch_log_events():
                if self._stop.is_set():
                    break
                if str(event.get("addr", "")).upper() != self._device:
                    continue
                data = bytes.fromhex(event.get("data_hex", ""))
                decoded = parse_capture_payload(data, self._tag)
                if decoded:
                    self._queue.put(decoded)
        except Exception as exc:  # surfaced on the next capture() get()
            self._queue.put(exc)

    def capture(
        self,
        timeout: float,
        retries: int,
        on_attempt: Callable[[int, int], None] | None = None,
    ) -> list[LH2CalibrationSample]:
        """Trigger one capture and return every station's record from the reply.

        Raises TimeoutError if nothing arrives within `retries + 1` triggers.
        """
        # Discard anything left over from the previous point.
        while not self._queue.empty():
            self._queue.get_nowait()

        attempts = retries + 1
        for attempt in range(attempts):
            if on_attempt is not None:
                on_attempt(attempt + 1, attempts)
            self._client.request_lh2_capture(self._device)
            deadline = time.monotonic() + timeout
            remaining = deadline - time.monotonic()
            while remaining > 0:
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(item, Exception):
                    raise item
                return item

        raise TimeoutError(
            f"no LH2 samples from {self._device} after {retries + 1} attempt(s); "
            f"is the DotBot in READY (app stopped) and in view of a lighthouse?"
        )

    def capture_point(
        self,
        point: int,
        reads: int = CAPTURE_READS_DEFAULT,
        timeout: float = CAPTURE_TIMEOUT_DEFAULT,
        retries: int = CAPTURE_RETRIES_DEFAULT,
        on_attempt: Callable[[int, int], None] | None = None,
        on_read: Callable[[int, int], None] | None = None,
    ) -> list[Sample]:
        """Take `reads` captures at one point and group them per station."""
        collected: list[list[LH2CalibrationSample]] = []
        for index in range(reads):
            if on_read is not None:
                on_read(index + 1, reads)
            collected.append(
                self.capture(timeout=timeout, retries=retries, on_attempt=on_attempt)
            )
        return samples_from_reads(collected, point)
