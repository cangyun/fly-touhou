"""Upper-field recovery using TH14's native resource snapshot."""

from ..contracts import ItemCollection
from ..item_collection import observation


def collection_state(snapshot):
    resources = snapshot.extra.get("resources", {})
    player = snapshot.player
    if player is None or not resources.get("observed"):
        return ItemCollection(None, None, None, None, None, None, None)
    line = resources.get("poc_line_y")
    playable = player.state == 1 and snapshot.screen == 7 and snapshot.pause == 2 and not snapshot.dialogue
    powers = [item for item in resources.get("items", ())
              if item.get("active", True) and item.get("kind") in (1, 3, 8)]
    attracting = sum(bool(item.get("autocollecting") or item.get("state") == 3) for item in powers)
    return observation(player_y=player.y, line_y=line, fast=playable, focused=playable,
                       player_focused=player.focused, waiting=len(powers) - attracting, attracting=attracting)
