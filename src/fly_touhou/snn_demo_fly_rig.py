"""Articulated 3D cartoon fly, baked into60FPS pose atlases for inexpensive playback.

This is presentation animation, not a biological locomotion simulation.
The body, two wings and six three-joint legs have independent transforms.
"""
import math
import numpy as np
from PySide6.QtCore import Qt,QPointF,QRectF
from PySide6.QtGui import QImage,QPainter,QPolygonF,QColor,QPen

WIDTH=420;HEIGHT=260;COUNT=40


def rx(a):
    c,s=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def ry(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def rz(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def ellipsoid(center,radius,colour,rings=12,sectors=24):
    p=np.linspace(.001,math.pi-.001,rings+1);t=np.arange(sectors)*math.tau/sectors
    u,v=np.meshgrid(p,t,indexing='ij')
    unit=np.stack([np.sin(u)*np.cos(v),np.sin(u)*np.sin(v),np.cos(u)],axis=-1).reshape(-1,3)
    vertices=np.asarray(center)+unit*np.asarray(radius)
    faces=[]
    for a in range(rings):
        for b in range(sectors):
            i=a*sectors+b;j=a*sectors+(b+1)%sectors
            faces.extend([(i,j,i+sectors),(j,j+sectors,i+sectors)])
    faces=np.array(faces);normals=(unit/np.asarray(radius))[faces].mean(axis=1)
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    return vertices[faces],normals,np.broadcast_to(colour,(len(faces),3)).copy()


class FlyRig:
    def __init__(self):
        objects=[ellipsoid((0,.64,.06),(.46,.63,.42),(230,161,72)),
                 ellipsoid((0,.05,.11),(.46,.57,.42),(247,185,94)),
                 ellipsoid((0,-.64,.16),(.57,.43,.47),(255,201,120))]
        for side in [-1,1]:
            objects.append(ellipsoid((side*.345,-1.0,.22),(.29,.235,.34),(235,54,38)))
            objects.append(ellipsoid((side*.345-.065,-1.21,.355),(.057,.025,.07),(255,246,230),6,12))
            objects.append(ellipsoid((side*.345+.065,-1.23,.24),(.023,.016,.029),(255,206,176),5,10))
        self.tri=np.concatenate([o[0] for o in objects]);self.norm=np.concatenate([o[1] for o in objects])
        self.col=np.concatenate([o[2] for o in objects]);self.camera=np.array([0,-.83,.56])
        self.depth_bias=np.concatenate([np.full(len(o[0]),0 if i<3 else .40) for i,o in enumerate(objects)])
        self.camera/=np.linalg.norm(self.camera);self.up=np.cross(self.camera,np.array([1,0,0]))
        self.light=np.array([-.4,-.65,.9]);self.light/=np.linalg.norm(self.light)

    def render(self,dx,dy,slow,phase,*,bomb=False):
        t=phase/60.
        pitch=math.radians(float(dy)*(8 if slow else 18))
        roll=math.radians(float(dx)*(8 if slow else 17))
        yaw=math.radians(float(dx)*(10 if slow else 23))
        if bomb:pitch-=.18;roll=0;yaw=0
        rotation=rz(yaw)@rx(pitch)@ry(roll)
        bob=(.018 if slow else .045)*math.sin(t*math.tau*(1.5 if slow else 3))
        scale=76 if slow else 79
        origin=np.array([WIDTH/2+float(dx)*13,HEIGHT/2+float(dy)*9+15])
        def project(x):
            world=np.asarray(x)@rotation.T+np.array([0,0,bob])
            return np.stack([world[...,0],-world@self.up],axis=-1)*scale+origin,world@self.camera
        normals=self.norm@rotation.T
        lighting=.49+.51*np.clip(normals@self.light,0,1)
        colours=np.clip(self.col*lighting[:,None],0,255).astype(int)
        visible=normals@self.camera>-.06
        xy,depth=project(self.tri)
        primitives=[]
        for i in np.flatnonzero(visible):
            primitives.append((float(depth[i].mean()+self.depth_bias[i]),'face',xy[i],tuple(colours[i])))
        def line(points,width,colour):
            xy,d=project(points);primitives.append((float(d.mean()),'line',xy,(width,colour)))
        # Segmented legs flex and sweep, with reduced excursions during focused movement.
        for side in [-1,1]:
            for leg,y in enumerate([-.30,.12,.49]):
                a=t*math.tau*(1.5 if slow else 3)+leg*math.pi/2+(math.pi if side<0 else 0)
                flex=(.035 if slow else .14)*math.sin(a)
                flare=.73 if slow else .87
                if bomb:flare=1.06;flex=.08
                points=np.array([[side*.32,y,-.08],[side*.58,y-.10,-.30],
                    [side*flare,y-.28+flex,-.58],[side*(flare+.09),y-.40+flex,-.72+abs(flex)*.5]])
                for k in range(3):line(points[k:k+2],5.6-k*.9,(208+11*k,143+10*k,66+9*k,255))
        for side in [-1,1]:
            line([[side*.13,-1.0,.12],[side*.22,-1.22,.16],[side*.34,-1.34,.05]],2.6,(173,117,59,255))
        # Wing membrane and veins share a hinge, rather than stretching the entire fly image.
        outline=np.array([[0,0,0],[.45,-.29,0],[1.10,-.39,0],[1.65,-.18,0],
            [1.82,.10,0],[1.76,.40,0],[1.40,.64,0],[.83,.61,0],[.23,.24,0]])
        for side in [-1,1]:
            flap=math.radians(3+(13 if slow else 26)*math.sin(t*math.tau*(6 if slow else 12)))
            if bomb:flap=math.radians(30-17*math.sin(t*math.tau*3))
            hinge=np.array([side*.25,.18,.29])
            local=outline.copy();local[:,0]*=side
            wing=local@ry(-side*flap).T+hinge
            smooth=[]
            for j in range(len(wing)):
                a,b,c,d=[wing[k%len(wing)] for k in [j-1,j,j+1,j+2]]
                for u in np.linspace(0,1,6,endpoint=False):
                    smooth.append(.5*((2*b)+(-a+c)*u+(2*a-5*b+4*c-d)*u*u+(-a+3*b-3*c+d)*u*u*u))
            wx,wd=project(smooth);primitives.append((float(wd.mean()),'wing',wx,None))
            for path in [[0,2,3],[0,6],[0,7],[1,7],[2,6],[3,5]]:
                line(wing[path],1.15,(205,220,226,145))
        # Tiny highlights suggest compound facets without turning the fly into a human face.
        for side in [-1,1]:
            for a in range(-2,3):
                for b in range(-2,3):
                    if a*a+b*b>5:continue
                    x=side*.345+a*.067;z=.22+b*.073
                    y=-1.0-.235*math.sqrt(max(.05,1-(a*.067/.29)**2-(b*.073/.34)**2))
                    xy,d=project([[x,y-.003,z]])
                    primitives.append((float(d[0]+.4),'dot',xy[0],(1.35,(180,35,27,110))))
        image=QImage(WIDTH*2,HEIGHT*2,QImage.Format_ARGB32_Premultiplied);image.fill(Qt.transparent)
        p=QPainter(image);p.scale(2,2)
        for _,kind,points,info in sorted(primitives,key=lambda x:x[0]):
            if kind=='face':
                p.setRenderHint(QPainter.Antialiasing,False)
                p.setPen(Qt.NoPen);p.setBrush(QColor(*map(int,info)))
                p.drawPolygon(QPolygonF([QPointF(float(x),float(y)) for x,y in points]))
            elif kind=='wing':
                p.setRenderHint(QPainter.Antialiasing,True)
                p.setPen(QPen(QColor(234,243,247,195),1.2));p.setBrush(QColor(204,225,234,150))
                p.drawPolygon(QPolygonF([QPointF(float(x),float(y)) for x,y in points]))
            elif kind=='line':
                p.setRenderHint(QPainter.Antialiasing,True)
                width,colour=info;p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor(*colour),width,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
                p.drawPolyline(QPolygonF([QPointF(float(x),float(y)) for x,y in points]))
            else:
                p.setRenderHint(QPainter.Antialiasing,True)
                radius,colour=info;p.setPen(Qt.NoPen);p.setBrush(QColor(*colour));p.drawEllipse(QPointF(*map(float,points)),radius,radius)
        p.end();return image.scaled(WIDTH,HEIGHT,Qt.IgnoreAspectRatio,Qt.SmoothTransformation)
