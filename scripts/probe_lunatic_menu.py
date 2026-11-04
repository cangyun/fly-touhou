from pathlib import Path
from datetime import datetime, timezone
import json
import traceback
from fly_touhou.frida_controller import OriginalGameController

root = Path(__file__).resolve().parents[1]
output = root/'artifacts/full_game/menu_probes'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output.mkdir(parents=True)
controller = OriginalGameController(
    r'games/th14/th14.exe',
    root/'profiles/th14_1.00b_original.json',timeout=25)
events = []
try:
    snapshot = controller.start()
    for attempt in range(180):
        row = {'attempt':attempt,'frame':snapshot.frame,'screen':snapshot.screen,
               'player':repr(snapshot.player),'extra':snapshot.extra}
        events.append(row)
        print('MENU',attempt,snapshot.screen,snapshot.extra.get('menu_page'),
              snapshot.extra.get('menu_substate'),snapshot.extra.get('menu_selection'),flush=True)
        if snapshot.player and snapshot.pause==2 and snapshot.screen==7:
            events.append({'started':snapshot.to_dict()})
            break
        if snapshot.extra.get('menu_page')==5:
            if snapshot.extra.get('menu_substate')!=2:
                snapshot=controller.advance(0,4)
                continue
            selection=snapshot.extra['menu_selection']
            if selection!=3:
                snapshot=controller.tap(0x20 if selection<3 else 0x10,4)
                continue
        snapshot=controller.tap(1,4)
        snapshot=controller.advance(0,8)
except BaseException:
    events.append({'error':traceback.format_exc()})
    if controller.reader:
        events.append({'raw_player':repr(controller.reader._player()),
                       'global':controller.reader.global_frame(),'stage':controller.reader.stage_frame()})
    raise
finally:
    controller.close()
    (output/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EVIDENCE',output,flush=True)
