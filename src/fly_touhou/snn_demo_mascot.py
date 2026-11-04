"""Decorative mascot driven by recorded chosen actions, never by argmax of marginals."""
import math,json
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QImage,QColor,QPen


class ActionMascot:
    def __init__(self,path):
        path=Path(path);self.atlases=None;self.rig=None
        if path.is_dir():
            self.rig=json.loads((path/'rig.json').read_text(encoding='utf-8'))
            self.atlases=[QImage(str(path/f'action-{a:02d}.png')) for a in range(19)]
            if any(im.isNull() for im in self.atlases):raise RuntimeError(f'Incomplete fly pose library: {path}')
        else:
            self.image=QImage(str(path))
            if self.image.isNull():raise RuntimeError(f'Mascot asset unavailable: {path}')
        self.bomb_frames=np.empty(0,np.int64);self.record_count=-1

    def update_records(self,rows):
        if len(rows)==self.record_count:return
        # One visual burst per contiguous Bomb selection. Its afterglow is decorative.
        self.bomb_frames=np.array([r['frame'] for i,r in enumerate(rows)
            if r['action']==18 and (i==0 or rows[i-1]['action']!=18)],np.int64)
        self.record_count=len(rows)

    def draw(self,p,rect,row,velocities,source_frame,video_frame):
        action=None if row is None else int(row['action'])
        bomb=action==18;slow=action is not None and 9<=action<18
        dx,dy=(0,0) if action is None or bomb else np.sign(velocities[action%9]).astype(int)
        t=video_frame/60.;cx=rect.center().x();cy=rect.top()+138
        colour=QColor('#ff789a' if bomb else '#67e8d2' if slow else '#ffcf79')
        p.save();p.setClipRect(rect.adjusted(6,6,-6,-6))
        p.setPen(Qt.NoPen);p.setBrush(QColor(33,53,68,95))
        p.drawEllipse(QRectF(cx-126,cy-100,252,200))
        if slow:
            p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor(103,232,210,145),2))
            p.drawEllipse(QPointF(cx,cy),109,109)
            for k in range(4):
                a=k*math.pi/2+t*.3
                p.drawLine(QPointF(cx+101*math.cos(a),cy+101*math.sin(a)),
                           QPointF(cx+119*math.cos(a),cy+119*math.sin(a)))
        if dx or dy:
            norm=math.hypot(dx,dy);ux,uy=dx/norm,dy/norm
            for k in range(5):
                phase=((t*(1.2 if slow else 3.1)+k/5)%1)
                distance=105+phase*70
                side=(k-2)*15
                x=cx-ux*distance-uy*side;y=cy-uy*distance+ux*side
                c=QColor(colour);c.setAlpha(int((1-phase)*120))
                p.setPen(QPen(c,2,Qt.SolidLine,Qt.RoundCap))
                p.drawLine(QPointF(x,y),QPointF(x-ux*(9 if slow else 23),y-uy*(9 if slow else 23)))
        i=int(np.searchsorted(self.bomb_frames,source_frame,side='right')-1)
        age=(source_frame-self.bomb_frames[i])/60 if i>=0 else 2
        if self.atlases is not None:
            a=18 if 0<=age<.3 else 0 if action is None else action
            n=min(self.rig['frames']-1,int(age*60)) if a==18 and age<.7 else video_frame%self.rig['frames']
            w,h=self.rig['width'],self.rig['height'];columns=self.rig['columns']
            p.drawImage(QRectF(cx-w/2,cy-h/2,w,h),self.atlases[a],QRectF(n%columns*w,n//columns*h,w,h))
        else:
            p.save();p.translate(cx+dx*17,cy+dy*11+(1.4 if slow else 3.5)*math.sin(t*math.tau*(1.8 if slow else 3.4)))
            p.rotate(dx*(5 if slow else 12)+dy*3);p.scale(-1 if dx<0 else 1,1)
            size=(292 if slow else 320)*(1+.006*math.sin(t*math.tau*8))
            p.drawImage(QRectF(-size/2,-size/2,size,size),self.image);p.restore()
        if 0<=age<.7:
            phase=age/.7;c=QColor('#ff789a');c.setAlpha(int((1-phase)*220))
            p.setBrush(Qt.NoBrush);p.setPen(QPen(c,4*(1-phase)+1))
            radius=65+phase*130;p.drawEllipse(QPointF(cx,cy),radius,radius)
            for k in range(8):
                a=math.tau*k/8
                p.drawLine(QPointF(cx+radius*math.cos(a),cy+radius*math.sin(a)),
                    QPointF(cx+(radius+15)*math.cos(a),cy+(radius+15)*math.sin(a)))
        p.restore()
