"""Same-frame metadata for TH14 callback42cb00's distance-reactive orbs."""
import struct

REVISION = 'th14-distance-reactive-orbs-v1'


def read_reactive_orbs(memory, snapshot, raw_bullets):
    resources = snapshot.extra.get('resources', {})
    contacts = [row['native_contact'] for row in
        (*resources.get('enemies', ()), *resources.get('hidden_enemy_contacts', ()))
        if row.get('native_contact', {}).get('special_tick_callback') == memory.base+0x2cb00]
    result = dict(revision=REVISION, bodies=[], points=[], programs=[],
                  player=[snapshot.player.x, snapshot.player.y])
    if not contacts:
        return result
    for contact in contacts:
        near, far = struct.unpack('<ff', memory.read(contact['data_address']+0x29c, 8))
        result['bodies'].append(dict(enemy_base=contact['enemy_base'], near_radius=near, far_radius=far))
    result['motion_time_rate'] = struct.unpack('<f', memory.read(memory.base+0xd8f58, 4))[0]
    programs = {}
    for raw in raw_bullets:
        if raw['native_state'] not in (1, 2):
            continue
        identity = raw['id']
        marker = struct.unpack('<i', memory.read(identity+0xbe8, 4))[0]
        if marker not in (1, 2):
            continue
        program = memory.read(identity+0xc28, 18*44).hex()
        index = programs.setdefault(program, len(programs))
        timer, _, cursor = struct.unpack('<iii', memory.read(identity+0xc18, 12))
        native = dict(raw, native_ex_index=cursor, native_motion_time_rate=result['motion_time_rate'])
        result['points'].append(dict(id=identity, marker=marker, reaction_timer=timer,
            program_index=index, native=native))
    result['programs'] = list(programs)
    return result
