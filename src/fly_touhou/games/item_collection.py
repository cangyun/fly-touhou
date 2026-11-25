"""Shared observation encoding for original-game automatic item recovery."""

import numpy as np

from .contracts import ItemCollection


SUFFIX = "-collection-v1"
NAMES = ("poc_can_trigger_now", "poc_fast_allowed_above_line",
         "poc_focused_allowed_above_line", "poc_attracting_power_items")
WIDTH = len(NAMES)


def input_schema(base, enabled):
    return base + SUFFIX if enabled else base


def validate(enabled):
    if type(enabled) is not bool:
        raise ValueError("item_collection_input must be a boolean")
    return enabled


def observation(*, player_y, line_y, fast, focused, player_focused, waiting, attracting):
    distance = line_y - player_y if line_y is not None else None
    allowed = focused if player_focused else fast
    current = bool(distance > 0. and allowed) if distance is not None and allowed is not None else None
    return ItemCollection(line_y, distance, current, fast, focused, waiting, attracting)


def augment(features, collection, *, field_height):
    """Use existing distance/count columns and append four distinct cues."""
    base = features.copy()
    distance = collection.distance_to_line
    base[128] = np.clip(distance / field_height, -1., 1.) if distance is not None else 0.
    count = collection.available_power_items
    if count is not None:
        base[118] = count / (count + 10.)
    attracting = collection.attracting_power_items
    tail = [collection.can_trigger_now, collection.fast_allowed_above_line,
            collection.focused_allowed_above_line,
            attracting / (attracting + 10.) if attracting is not None else None]
    return np.concatenate((base, np.asarray([-1. if value is None else value for value in tail], np.float32)))
