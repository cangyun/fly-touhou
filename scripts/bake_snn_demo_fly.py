from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage,QPainter,QColor
from PySide6.QtCore import QRectF
from fly_touhou.snn_demo_fly_rig import FlyRig,WIDTH,HEIGHT,COUNT


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    app=QApplication(sys.argv);rig=FlyRig()
    velocities=json.loads((args.recording/'manifest.json').read_text(encoding='utf-8'))['velocities']
    if args.preview:
        image=QImage(WIDTH*3,HEIGHT*2,QImage.Format_ARGB32_Premultiplied);image.fill(QColor('#0e1825'))
        paint=QPainter(image)
        for i,(dx,dy,slow,bomb) in enumerate([(0,0,False,False),(-1,0,False,False),(1,0,False,False),
                (0,-1,False,False),(0,1,True,False),(0,0,False,True)]):
            paint.drawImage(QRectF(i%3*WIDTH,i//3*HEIGHT,WIDTH,HEIGHT),rig.render(dx,dy,slow,2,bomb=bomb))
        paint.end();image.save(str(args.output/'rig-preview.png'));return
    for action in range(19):
        v=velocities[action%9] if action<18 else [0,0]
        dx=0 if v[0]==0 else 1 if v[0]>0 else -1;dy=0 if v[1]==0 else 1 if v[1]>0 else -1
        atlas=QImage(WIDTH*10,HEIGHT*4,QImage.Format_ARGB32_Premultiplied);atlas.fill(QColor(0,0,0,0))
        paint=QPainter(atlas)
        for frame in range(COUNT):paint.drawImage(QRectF(frame%10*WIDTH,frame//10*HEIGHT,WIDTH,HEIGHT),rig.render(dx,dy,9<=action<18,frame,bomb=action==18))
        paint.end();atlas.save(str(args.output/f'action-{action:02d}.png'));print('baked',action,flush=True)
    (args.output/'rig.json').write_text(json.dumps(dict(width=WIDTH,height=HEIGHT,frames=COUNT,columns=10,fps=60,
        articulation='3D body orientation,two independent wing hinges,six articulated legs;decorative animation',
        action_source='actual recorded action index;19 actions',probability_changes=False),indent=2),encoding='utf-8')


if __name__=='__main__':main()
