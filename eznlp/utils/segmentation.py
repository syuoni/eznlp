# -*- coding: utf-8 -*-
import re
from typing import List


def segment_text_with_seps(text: str, seps: List[str], length: int = None):
    """Segment text with a list of separators.

    Notes
    -----
    (1) If `length` is not provided (`None` or `0`), it will segment at all possible seperators.
    (2) If `length` is provided, the resulting spans will be as long as possible to be close to `length`.
    A span may exceed `length`, if the corresponding distance between two successive separators exceeds it.
    """
    if length is None:
        length = 0

    start, end = 0, None
    for cut in re.finditer("|".join(seps), text):
        if cut.end() - start <= length:
            end = cut.end()

        elif end is None:
            yield (start, cut.end())
            start = cut.end()

        else:
            yield (start, end)

            if cut.end() - end <= length:
                start, end = end, cut.end()
            else:
                yield (end, cut.end())
                start, end = cut.end(), None

    if len(text) > start:
        if len(text) - start <= length or end is None:
            yield (start, len(text))
        else:
            yield (start, end)
            yield (end, len(text))


def segment_text_with_hierarchical_seps(
    text: str, hie_seps: List[List[str]], length: int = None
):
    """Segment text with hierarchical lists of separators.

    Notes
    -----
    Segment text first with seperators in `hie_seps[0]`. For the spans longer than `length`,
    further segment the spans with separators in `hie_seps[1]`, and so on.
    """
    if len(hie_seps) == 0 or sum(len(seps) for seps in hie_seps) == 0:
        yield (0, len(text))

    else:
        for start, end in segment_text_with_seps(text, hie_seps[0], length=length):
            if length is not None and end - start <= length:
                yield (start, end)
            else:
                for sub_start, sub_end in segment_text_with_hierarchical_seps(
                    text[start:end], hie_seps[1:], length=length
                ):
                    # Add offset to the spans from sub-spans
                    yield (start + sub_start, start + sub_end)


def segment_text_uniformly(text: str, num_spans: int = None, max_span_size: int = None):
    assert not (num_spans is None and max_span_size is None)

    if num_spans is None:
        num_spans, tail = divmod(len(text), max_span_size)
        if tail > 0:
            num_spans += 1

    span_size = len(text) / num_spans
    for i in range(num_spans):
        start = int(span_size * i + 0.5)
        end = int(span_size * (i + 1) + 0.5)
        yield (start, end)
