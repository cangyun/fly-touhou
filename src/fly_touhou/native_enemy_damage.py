"""Original TH14 1.00b incoming-damage bound, not an empirical DPS estimate.

4516dd..4516eb clamps the combined shot/Bomb result of451430 to
Player(+688)->shot_definition(+28).423a86..423aae can add an enemy callback
and a queued one-off amount AFTER that cap.424eb0/424f90/425070/425180 can
apply99999 damage to enemies with flag800. Those exceptional paths must not
be silently covered by the ordinary shot cap. This module only reads memory.
"""
import struct


REVISION = 'th14-native-shot-bomb-damage-bound-v1'


def read_damage_bound(memory, data, image_base, flags):
    def word(address, signed=False):
        return struct.unpack('<i' if signed else '<I', memory.read(address, 4))[0]
    answer = dict(revision=REVISION, per_frame=None, initial_extra=0,
                  source='native451430 cap;423a86 extra sources;424eb0 flag800 exception')
    player = word(image_base + 0xdb67c)
    if not player:
        return dict(answer, reason='player unavailable')
    definition = word(player + 0x688)
    if not definition:
        return dict(answer, reason='shot definition unavailable')
    cap = word(definition + 0x28, True)
    callback = word(data + 0x4114)
    extra = max(0, word(data + 0x3fd0, True))
    answer.update(shot_definition=definition, shot_bomb_cap=cap,
                  extra_damage_callback=callback, initial_extra=extra,
                  exceptional_area_damage=bool(flags & 0x800))
    if callback:
        return dict(answer, reason='enemy extra-damage callback is not bounded')
    if flags & 0x800:
        return dict(answer, reason='native99999 area-damage path is eligible')
    if cap < 0:
        return dict(answer, reason='negative native damage cap')
    # Subsequent rank/resistance/spell factors in423ac2..423b9b only reduce
    # damage;421480 health conversion also cannot exceed the raw damage.
    return dict(answer, per_frame=cap, reason=None)
