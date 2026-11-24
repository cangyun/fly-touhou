"""Read-only TH14 ECL role evidence at an already-owned input checkpoint.

No attach/launch/write/hook, no native-module import, no defeat inference.
Offset semantics checked against profiles/th14_1.00b_original.json and
ParaKit state_reader.py get_sub_name_from_ref/extract_enemies/find_section_sub
(especially its TH14 MainBoss + MBossCard/Seija exception). This is a separate
conservative implementation: an M prefix or subboss_id alone proves no role.
"""
from __future__ import annotations

import bisect
import struct


SCHEMA = 'th14-thrl-ecl-roles-v1'


def read_enemy_ecl_roles(controller, snapshot):
    """Return current primary midboss HP/identity only when roles are resolved.

    Caller supplies a paused, owned controller and its same-frame Snapshot.
    NativeReader normally removes invisible nonboss ECL controller entities;
    this helper must therefore inspect the manager's *full* list, including
    those entities. It never assumes Snapshot.enemies contains MainMBoss.

    On ambiguity/read failure: midboss_observed=False, no synthetic empty or
    defeat event. A successful empty list means current absence only, NOT kill.
    """
    result = {'schema': SCHEMA, 'observed': False, 'midboss_observed': False,
              'midbosses': [], 'main_bosses': [], 'auxiliary_bosses': [],
              'main_enemy_subs': [], 'main_thread_role': 'unknown', 'enemies': [],
              'snapshot_frame': int(snapshot.frame), 'errors': [],
              'absence_is_defeat': False,
              'input_contract': 'existing owned input checkpoint; no attach or writes',
              'ecl_ref_kind': 'TH14 current instruction pointer, not TH15+ serial sub id'}
    try:
        snapshot.validate()
        profile = controller.profile
        if profile.game_id != 14:
            raise ValueError('Only pinned TH14 profile semantics are supported')
        memory = controller._memory
        if memory is None:
            raise ValueError('Controller has no existing process memory')
        offsets = profile.offsets['enemies']
        misc = profile.offsets['misc']

        def integer(address, *, signed=False, size=4):
            return int(memory.integer(address, signed=signed, size=size))

        def frame():
            ascii_manager = integer(memory.base+misc['ascii_manager_ptr'])
            if not ascii_manager:
                raise ValueError('Missing native global frame manager')
            return integer(ascii_manager+misc['global_timer'])

        before = frame()
        if before != snapshot.frame:
            raise ValueError('Role read did not begin at its Snapshot frame')
        manager = integer(memory.base+offsets['enemy_manager_pointer'])
        if not manager:
            raise ValueError('Enemy manager is absent; role absence is unobserved')
        ecl_file = integer(manager+offsets['zEnemyManager_ecl_file'])
        if not ecl_file:
            raise ValueError('Enemy ECL file is absent')
        count = integer(ecl_file+offsets['zEclFile_sub_count'])
        table = integer(ecl_file+offsets['zEclFile_subroutines'])
        if not 1 <= count <= 8192 or not table:
            raise ValueError('ECL subroutine table has invalid/missing extent')
        # One complete table RPM remains mandatory at each checkpoint. Only
        # its stage-lifetime names/index are reusable, never live enemy state.
        table_bytes = memory.read(table, count*8)
        cache_key = (id(memory), getattr(memory, 'pid', getattr(controller, 'pid', None)),
                     int(snapshot.stage), manager, ecl_file, table, count, table_bytes)
        cache = getattr(controller, '_thrl_ecl_subroutine_cache', None)
        if cache is None or cache['key'] != cache_key or cache['memory'] is not memory:
            controller._thrl_ecl_subroutine_cache = None
            subroutines = []
            names_by_start = {}
            for name_pointer, start in struct.iter_unpack('<II', table_bytes):
                if not name_pointer or not start:
                    raise ValueError('Incomplete ECL subroutine entry')
                raw = memory.read(name_pointer, 64)
                if b'\0' not in raw:
                    raise ValueError('ECL name exceeds the bounded64-byte read')
                name = raw.split(b'\0', 1)[0].decode('ascii', errors='strict')
                subroutines.append((start, name))
                names_by_start.setdefault(start, set()).add(name)
            subroutines.sort()
            cache = {'key': cache_key, 'memory': memory,
                     'starts': tuple(entry[0] for entry in subroutines),
                     'names_by_start': {address: frozenset(names) for address, names in names_by_start.items()}}
            controller._thrl_ecl_subroutine_cache = cache
        starts, names_by_start = cache['starts'], cache['names_by_start']

        def sub_name(reference):
            if reference in (0, 0xffffffff):
                return None, 'invalid_or_initializing_native_ecl_ref'
            index = bisect.bisect_right(starts, reference)-1
            if index < 0:
                return None, 'before_first_sub_or_initializing'
            start = starts[index]
            names = names_by_start[start]
            if len(names) != 1:
                return None, 'aliased_sub_start_ambiguous'
            # This is the actual native current-ECL-instruction field, not an
            # arbitrary address. ParaKit's TH14 mapper uses greatest start<=ref
            # for the final allocated sub too. Heap allocation order can place
            # the live stage main last; requiring a next start falsely rejects
            # it. Keep the absent independent upper extent explicit in evidence.
            if index == len(starts)-1 and reference != start:
                return next(iter(names)), 'native_ecl_greatest_start_last_extent_unverified'
            return next(iter(names)), 'exact_start' if reference == start else 'between_verified_sub_starts'

        head = integer(manager+offsets['zEnemyManager_list'])
        seen, addresses, selected = set(), set(), set()
        while head:
            if head in seen or len(seen) >= 2048:
                raise ValueError('Native enemy list is cyclic or exceeds the explicit2048-node bound')
            seen.add(head)
            enemy, head = struct.unpack('<II', memory.read(head, 8))
            if not enemy:
                continue
            if enemy in addresses:
                raise ValueError('Duplicate native enemy identity')
            addresses.add(enemy)
            data = enemy+offsets['zEnemy_data']
            flags = integer(data+offsets['zEnemyData_flags'], size=8)
            is_boss = bool(flags & offsets['zEnemyFlags_is_boss'])
            intangible = bool(flags & offsets['zEnemyFlags_intangible'])
            anm = integer(data+offsets['zEnemyData_anm_vm_id'])
            reference = integer(enemy+offsets['zEnemy_ecl_ref'])
            name, resolution = sub_name(reference)
            hp = integer(data+offsets['zEnemyData_hp'], signed=True)
            row = {'id': enemy, 'hp': hp,
                   'hp_max': integer(data+offsets['zEnemyData_hp_max'], signed=True),
                   'subboss_id': integer(data+offsets['zEnemyData_subboss_id'], signed=True),
                   'boss': is_boss, 'intangible': intangible, 'anm_vm_id': anm,
                   'ecl_ref': reference, 'ecl_sub_name': name, 'ecl_resolution': resolution}
            result['enemies'].append(row)
            if is_boss or anm:
                selected.add(enemy)
        # Same filter as NativeReader._enemies, but keep hidden controller rows
        # above for ECL context. No guessed Main* entry from the visible subset.
        expected = {enemy.id: enemy for enemy in snapshot.enemies}
        if selected != set(expected):
            raise ValueError('Native full-list visible identities do not match Snapshot')
        for row in result['enemies']:
            if row['id'] in expected:
                enemy = expected[row['id']]
                if (row['hp'], row['boss'], row['intangible']) != (enemy.hp, enemy.boss, enemy.intangible):
                    raise ValueError('Native role HP/flags disagree with Snapshot')
        if (frame() != before or integer(memory.base+offsets['enemy_manager_pointer']) != manager
                or integer(manager+offsets['zEnemyManager_ecl_file']) != ecl_file):
            raise ValueError('Role read crossed a native frame/manager/ECL lifetime')
        result.update(observed=True, native_frame_before=before, native_frame_after=before,
                      ecl_subroutine_count=count, complete_native_enemy_nodes=len(seen))
        main_rows = [row for row in result['enemies'] if row['ecl_sub_name']
                     and row['ecl_sub_name'].lower().startswith('main')]
        result['main_enemy_subs'] = [row['ecl_sub_name'] for row in main_rows]
        families = set()
        for row in main_rows:
            name = row['ecl_sub_name']
            if name.startswith('MainMBoss'):
                families.add('midboss')
            elif name.startswith('MainBoss'):
                families.add('main_boss')
        candidates = [row for row in result['enemies'] if row['boss']]
        primary = [row for row in candidates if row['subboss_id'] == 0]
        result['auxiliary_bosses'] = [row for row in candidates if row['subboss_id'] != 0]
        if not candidates and len(families) <= 1:
            # Full same-frame list absence is measurable even while the Main*
            # controller remains alive. It is not evidence of a native kill.
            result.update(midboss_observed=True, main_thread_role='no_current_boss')
            return result
        if len(families) > 1:
            result['errors'].append('Conflicting MainBoss/MainMBoss context; no M-prefix-only inference')
            return result
        if len(primary) != 1:
            result['errors'].append('Primary native boss identity is missing or ambiguous')
            return result
        family = next(iter(families), None)
        evidence = 'unique MainBoss/MainMBoss context'
        if family is None:
            # Observed TH14 s1 first spawn: the complete same-ECL list has
            # lowercase "main" and a unique real primary boss in sub "MBoss".
            # ParaKit accepts main* stage threads and strips Seija's M only
            # under explicit MainBoss. Do not require MainMBoss to exist.
            # Same rule as ParaKit extract_enemies: resolved same-ECL main*
            # threads include MainLatter and other stage phases. A finite name
            # whitelist falsely rejected four real s1 MainLatter/MBossCard1
            # observations. Explicit role families above remain authoritative.
            stage_context = bool(main_rows)
            phase = primary[0]['ecl_sub_name'] or ''
            if stage_context and phase.startswith('MBoss'):
                family = 'midboss'
            elif stage_context and phase.startswith('Boss'):
                family = 'main_boss'
            else:
                result['errors'].append('No resolved explicit role or stage-context + MBoss/Boss stem; no M-prefix-only inference')
                return result
            evidence = 'resolved same-ECL stage main-prefix context (ParaKit) + complete MBoss/Boss sub stem'
        result['main_thread_role'] = family
        # ParaKit's canonical phase regex and Hide exclusion select display
        # section names; they are not role-observation requirements. The actual
        # primary identity/HP remain measured through utility/Hide routines.
        chosen = dict(primary[0], role=family, role_evidence=evidence+' + unique native boss identity/subboss_id0 + measured HP')
        # MainBoss + MBossCard3 remains a main boss (Seija), not midboss.
        if family == 'midboss':
            result['midbosses'] = [chosen]
        else:
            result['main_bosses'] = [chosen]
        result['midboss_observed'] = True
        return result
    except Exception as error:
        # Do not carry a cache obtained during an inconsistent/read-failed
        # checkpoint into the next observation. New runs also clear this attr.
        controller._thrl_ecl_subroutine_cache = None
        result['observed'] = False
        result['midboss_observed'] = False
        result['midbosses'] = []
        result['main_bosses'] = []
        result['errors'].append(type(error).__name__+': '+str(error))
        return result
