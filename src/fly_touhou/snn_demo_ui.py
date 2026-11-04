"""Native Qt recording canvas: original game, actual SNN activity and actions."""
from pathlib import Path
import json,time,subprocess
import numpy as np
import cv2
from PySide6.QtCore import Qt,QTimer,QRectF,QPointF,QUrl
from PySide6.QtGui import QPainter,QColor,QFont,QImage,QPen,QPolygonF,QFontDatabase
from PySide6.QtMultimedia import QMediaPlayer,QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QApplication,QWidget,QMainWindow,QVBoxLayout,QHBoxLayout,QPushButton,QSlider,QLabel,QFileDialog,QStackedWidget,QComboBox
from .snn_demo_data import TraceReader
from .snn_demo_mascot import ActionMascot
from .thrl_study_io import read_shared_json

BG='#080e18';PANEL='#0e1825';LINE='#20364a';TEXT='#e9f3fa';MUTED='#7f9cb1'
CYAN='#67e8d2';GOLD='#ffcf79';PINK='#ff789a'


def qimage(rgb):
    rgb=np.ascontiguousarray(rgb,dtype=np.uint8)
    return QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()


class BrainImage:
    def __init__(self,atlas):
        xyz=atlas['xyz_um'].astype(np.float32);self.valid=np.isfinite(xyz).all(1)
        xyz=xyz[self.valid];xyz-=np.median(xyz,axis=0)
        xyz/=np.max(np.ptp(xyz,axis=0))*.52
        self.xyz=xyz;self.cached=None;self.key=None
    def draw(self,rate,frame):
        key=(frame//4,id(rate))
        if self.key==key:return self.cached
        # Camera motion is geometric only. Activity always comes from the trace.
        yaw=.16*np.sin(frame/1800);c,s=np.cos(yaw),np.sin(yaw)
        x=self.xyz[:,0]*c+self.xyz[:,2]*s
        y=self.xyz[:,1]*.90-self.xyz[:,2]*.35
        w,h=440,260
        px=np.clip(((x*.43+.5)*w).astype(int),0,w-1)
        py=np.clip(((y*.76+.5)*h).astype(int),0,h-1)
        index=py*w+px
        base=np.bincount(index,minlength=w*h).reshape(h,w).astype(np.float32)
        signal=np.zeros(len(self.xyz),np.float32) if rate is None else np.sqrt(np.clip(rate[self.valid],0,1))
        active=np.bincount(index,weights=signal,minlength=w*h).reshape(h,w).astype(np.float32)
        anatomy=np.clip(np.log1p(base)*.12,0,.46)
        spots=1-np.exp(-active*.7)
        glow=cv2.GaussianBlur(spots,(0,0),1.6)
        rgb=np.empty((h,w,3),np.float32);rgb[:]=[14,24,37]
        rgb+=anatomy[...,None]*np.array([45,105,145])
        rgb+=spots[...,None]*np.array([75,185,142])+glow[...,None]*np.array([70,120,70])
        self.cached=qimage(np.clip(rgb,0,255).astype(np.uint8));self.key=key
        return self.cached


class DemoCanvas(QWidget):
    def __init__(self,recording,media_path=None):
        super().__init__();self.recording=Path(recording);self.media_path=media_path
        self.trace=TraceReader(recording)
        self.manifest=json.loads((self.recording/'manifest.json').read_text(encoding='utf-8'))
        self.atlas=np.load(self.recording/'atlas.npz');self.brain=BrainImage(self.atlas)
        self.groups=self.atlas['groups'];self.group_counts=[int((self.groups==g).sum()) for g in range(3)]
        self.frame=0;self.row=None;self.rate=None
        settings=self.recording/'presentation-options.json'
        options=json.loads(settings.read_text(encoding='utf-8')) if settings.exists() else {}
        asset=options.get('mascot_rig',options.get('mascot_image'))
        self.mascot=ActionMascot(asset) if asset else None
        self.source_frame=0
        self.video=None;self.video_count=0;self.video_index=-1;self.game_image=None;self.frame_map=None
        self.setMinimumSize(960,540);self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.open_video();self.set_frame(0)
    def open_video(self):
        media={};manifest=Path(self.media_path) if self.media_path else self.recording/'presentation-media.json'
        if manifest.exists():media=json.loads(manifest.read_text(encoding='utf-8'))
        self.media=media;self.video_frame_offset=int(media.get('video_frame_offset',0))
        audio=media.get('audio')
        self.audio_path=Path(audio) if isinstance(audio,str) and Path(audio).is_file() else None
        path=Path(media.get('video',self.recording/'game.mp4'))
        if path.exists():
            cap=cv2.VideoCapture(str(path))
            if cap.isOpened():
                self.video=cap;self.video_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                m=Path(media.get('frame_map',self.recording/'video-source-frames.npy'))
                if m.exists():self.frame_map=np.load(m)
    @property
    def duration(self):
        return max(1,self.video_count if self.video else (int(self.trace.frames[-1])+4 if len(self.trace.frames) else 1))
    def set_frame(self,frame):
        self.frame=int(np.clip(frame,0,self.duration-1))
        source=self.frame if self.frame_map is None else int(self.frame_map[min(self.frame,len(self.frame_map)-1)])
        self.source_frame=source
        self.row,self.rate=self.trace.at(source)
        if self.mascot is not None:self.mascot.update_records(self.trace.rows)
        if self.video is not None and self.frame!=self.video_index:
            delta=self.frame-self.video_index
            if 1<delta<=8:
                for _ in range(delta-1):self.video.grab()
            elif delta!=1:self.video.set(cv2.CAP_PROP_POS_FRAMES,self.frame)
            ok,bgr=self.video.read()
            if ok:self.game_image=qimage(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));self.video_index=self.frame
        self.update()
    def text(self,p,x,y,w,h,value,size=20,color=TEXT,bold=False,align=Qt.AlignLeft):
        p.setPen(QColor(color));font=QFont('Microsoft YaHei',size)
        if bold:font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font);p.drawText(QRectF(x,y,w,h),int(align|Qt.AlignVCenter),str(value))
    def box(self,p,x,y,w,h,fill=PANEL,border=LINE,radius=16):
        p.setPen(QPen(QColor(border),1));p.setBrush(QColor(fill));p.drawRoundedRect(QRectF(x,y,w,h),radius,radius)
    def bar(self,p,x,y,w,value,color):
        p.setPen(Qt.NoPen);p.setBrush(QColor('#1c2c3a'));p.drawRoundedRect(QRectF(x,y,w,12),6,6)
        p.setBrush(QColor(color));p.drawRoundedRect(QRectF(x,y,max(0,min(1,float(value)))*w,12),6,6)
    def paintEvent(self,event):
        p=QPainter(self);self.paint_scene(p,self.width(),self.height());p.end()
    def paint_scene(self,p,width,height):
        p.setRenderHint(QPainter.Antialiasing);p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(0,0,width,height,QColor(BG));scale=min(width/1920,height/1080)
        p.translate((width-1920*scale)/2,(height-1080*scale)/2);p.scale(scale,scale)
        self.text(p,1510,28,375,44,'LUNATIC  ·  REIMU A',20,GOLD,True,Qt.AlignRight)
        self.box(p,32,24,464,292);self.box(p,32,332,464,678)
        self.box(p,520,120,1024,768);self.box(p,1568,120,320,890)
        if self.mascot is not None:
            self.mascot.draw(p,QRectF(32,24,464,292),self.row,np.array(self.manifest['velocities']),self.source_frame,self.frame+self.video_frame_offset)
        self.text(p,56,347,400,40,'神经活动',22,TEXT,True)
        self.text(p,56,387,410,28,f'{len(self.groups):,} 个神经元  ·  完整连接组',13,MUTED)
        p.drawImage(QRectF(44,416,440,260),self.brain.draw(self.rate,self.frame+self.video_frame_offset))
        self.text(p,58,680,396,24,'低活动',11,MUTED)
        for k in range(64):
            t=k/63;p.fillRect(QRectF(156+k*3.2,688,3.3,7),QColor(int(30+100*t),int(65+175*t),int(75+100*t)))
        self.text(p,375,680,90,24,'高活动',11,CYAN,align=Qt.AlignRight)
        row=self.row
        self.text(p,58,716,408,23,'神经元数量  /  平均活动强度',11,MUTED)
        means=[0,0,0] if row is None else row['population_rate']
        for j,(group,title,color) in enumerate([(1,'感觉输入',CYAN),(0,'脑内神经群','#8aaee8'),(2,'下行输出',GOLD)]):
            y=742+j*68
            self.text(p,58,y,210,30,title,16,TEXT,True)
            self.text(p,283,y,184,30,f'{self.group_counts[group]:,}  /  {means[group]*100:.1f}%',13,color,align=Qt.AlignRight)
            self.bar(p,58,y+31,408,means[group],color)
        self.text(p,58,939,404,26,'群体活动轨迹',12,MUTED)
        if row is not None:
            history=[r for r in self.trace.rows[max(0,row['index']-120):row['index']+1]]
            for group,color in [(0,'#8aaee8'),(1,CYAN),(2,GOLD)]:
                values=[r['population_rate'][group] for r in history]
                if len(values)>1:
                    max_value=max(.05,max(max(r['population_rate']) for r in history))
                    points=[QPointF(58+408*i/max(1,len(values)-1),995-v/max_value*28) for i,v in enumerate(values)]
                    p.setPen(QPen(QColor(color),2));p.setBrush(Qt.NoBrush);p.drawPolyline(QPolygonF(points))
        if self.game_image is not None:p.drawImage(QRectF(520,120,1024,768),self.game_image)
        else:
            self.text(p,520,400,1024,70,'原作画面采集中',26,MUTED,align=Qt.AlignCenter)
            self.text(p,520,474,1024,38,'游戏画面将按原作帧与脑活动对齐',16,MUTED,align=Qt.AlignCenter)
        self.box(p,520,912,1024,98)
        action=0 if row is None else row['action']
        joint=np.zeros(19) if row is None else np.array(row['probabilities']);mass=joint[:18].sum()
        cond=joint[:18]/mass if mass>0 else np.zeros(18);direction=cond[:9]+cond[9:]
        slow=cond[9:].sum();bomb=joint[18]
        velocity=np.array(self.manifest['velocities']);d=0 if action==18 else action%9
        signs=np.sign(velocity[:9]).astype(int)
        names={(0,0):'停留',(-1,0):'左',(1,0):'右',(0,-1):'上',(0,1):'下',(-1,-1):'左上',(1,-1):'右上',(-1,1):'左下',(1,1):'右下'}
        arrows={(0,0):'·',(-1,0):'←',(1,0):'→',(0,-1):'↑',(0,1):'↓',(-1,-1):'↖',(1,-1):'↗',(-1,1):'↙',(1,1):'↘'}
        chosen='Bomb' if action==18 else names[tuple(signs[d])]+(' · 低速' if action>=9 else ' · 高速')
        game_time=(self.frame+self.video_frame_offset)/60;mm=int(game_time//60);ss=int(game_time%60)
        self.text(p,546,930,270,52,f'{mm:02d}:{ss:02d}',28,GOLD,True)
        self.text(p,772,928,300,56,f'第 {row["stage"] if row else "—"} 面',24,TEXT,True)
        self.text(p,1060,928,455,56,chosen,24,CYAN,True,Qt.AlignRight)
        self.text(p,1592,140,270,40,'方向选择',22,TEXT,True)
        for i,(dx,dy) in enumerate(signs):
            x=1590+(dx+1)*94;y=240+(dy+1)*96
            selected=action!=18 and d==i
            fill='#20483f' if selected else '#142332';border=CYAN if selected else LINE
            self.box(p,x,y,88,88,fill,border,10)
            self.text(p,x,y+2,88,45,arrows[(dx,dy)],27,CYAN if selected else TEXT,True,Qt.AlignCenter)
            self.text(p,x,y+48,88,30,f'{direction[i]*100:.1f}%',13,GOLD if selected else MUTED,align=Qt.AlignCenter)
        self.text(p,1592,564,270,34,'低速动作总概率',18,TEXT,True)
        self.text(p,1592,603,270,55,f'{slow*100:.1f}%',34,CYAN,True)
        self.bar(p,1592,673,272,slow,CYAN)
        low=action<18 and action>=9
        self.text(p,1592,700,270,30,'已选择低速' if low else '未选择低速',14,CYAN if low else MUTED)
        self.text(p,1592,770,270,34,'Bomb',20,TEXT,True)
        self.text(p,1592,811,270,56,f'{bomb*100:.1f}%',34,PINK,True)
        self.bar(p,1592,882,272,bomb,PINK)
        self.text(p,1592,909,270,32,'Bomb 已选择' if action==18 else '待命',15,PINK if action==18 else MUTED)
        self.text(p,32,1030,1380,25,'FlyWire FAFB v783  ·  感觉输入 → 全脑脉冲活动 → 下行动作读出',13,MUTED)
        self.text(p,1490,1030,398,25,'TH14  /  60 FPS',13,MUTED,align=Qt.AlignRight)
    def image(self):
        im=QImage(1920,1080,QImage.Format_RGB888);p=QPainter(im);self.paint_scene(p,1920,1080);p.end();return im


class DemoWindow(QMainWindow):
    def __init__(self,recording,presentation=None):
        super().__init__();self.setWindowTitle('果蝇 × 东方 · 演示与录制')
        self.presentation=presentation;self.last_status_poll=0.
        self.setAttribute(Qt.WA_ShowWithoutActivating);self.resize(1440,875)
        self.canvas=DemoCanvas(recording);self.playing=False;self.origin=0.;self.first=0
        self.audio_output=QAudioOutput(self);self.audio_output.setVolume(.8)
        self.audio_player=QMediaPlayer(self);self.audio_player.setAudioOutput(self.audio_output)
        self.direct_playback=False;self.playback_frames=self.canvas.duration
        self.last_audio_sync=0.
        central=QWidget();self.setCentralWidget(central);layout=QVBoxLayout(central);self.layout=layout;layout.setContentsMargins(0,0,0,0)
        self.stack=QStackedWidget();self.stack.addWidget(self.canvas)
        self.video_widget=QVideoWidget();self.video_widget.setAspectRatioMode(Qt.KeepAspectRatio);self.stack.addWidget(self.video_widget)
        self.video_widget.videoSink().videoFrameChanged.connect(self.video_frame_ready)
        layout.addWidget(self.stack,1);self.controls=QWidget();line=QHBoxLayout(self.controls);layout.addWidget(self.controls)
        play=QPushButton('播放 / 暂停');play.clicked.connect(self.toggle);line.addWidget(play)
        self.slider=QSlider(Qt.Horizontal);self.slider.setRange(0,self.canvas.duration-1);self.slider.sliderMoved.connect(self.seek);line.addWidget(self.slider,1)
        save=QPushButton('保存画面');save.clicked.connect(self.save_frame);line.addWidget(save)
        refresh=QPushButton('刷新录像');refresh.clicked.connect(self.refresh);line.addWidget(refresh)
        self.chapters=QComboBox();self.chapters.setToolTip('跳到对应关卡')
        self.chapters.activated.connect(lambda index:self.seek(int(self.chapters.itemData(index) or 0)));line.addWidget(self.chapters)
        self.music_label=QLabel('音量');line.addWidget(self.music_label)
        volume=QSlider(Qt.Horizontal);volume.setRange(0,100);volume.setValue(80);volume.setFixedWidth(85)
        volume.setToolTip('总音量：音乐与原作音效');volume.valueChanged.connect(lambda value:self.audio_output.setVolume(value/100))
        line.addWidget(volume)
        self.audio_player.errorOccurred.connect(self.audio_error)
        self.open_audio()
        seconds=self.canvas.duration//60
        self.status_label=QLabel(f'可播放 {seconds//60:02d}:{seconds%60:02d}');line.addWidget(self.status_label)
        clean=QPushButton('录制视图 · F11');clean.clicked.connect(self.clean);line.addWidget(clean)
        self.setStyleSheet('QMainWindow,QWidget{background:#080e18;color:#d7e4ee} QPushButton{background:#1c3446;border:0;padding:10px 18px;border-radius:6px}')
        self.timer=QTimer(self);self.timer.setTimerType(Qt.PreciseTimer);self.timer.timeout.connect(self.tick);self.timer.start(16)
    def refresh(self):
        self.playing=False
        self.audio_player.stop()
        selected=self.canvas.recording
        if self.presentation is not None:selected=Path(read_shared_json(self.presentation)['recording'])
        if selected!=self.canvas.recording:
            old=self.canvas;self.canvas=DemoCanvas(selected);self.stack.removeWidget(old);self.stack.insertWidget(0,self.canvas)
            if old.video is not None:old.video.release()
            old.deleteLater()
            self.open_audio();self.slider.setRange(0,self.playback_frames-1);self.seek(0);return
        self.canvas.trace.refresh()
        if self.canvas.video is not None:self.canvas.video.release()
        self.canvas.video=None;self.canvas.video_index=-1;self.canvas.open_video()
        self.open_audio();self.slider.setRange(0,self.playback_frames-1);self.seek(0)
    def open_audio(self):
        self.stack.setCurrentWidget(self.canvas)
        composed=self.canvas.media.get('presentation_video')
        if self.presentation is not None:
            selected=read_shared_json(self.presentation)
            if not composed and selected.get('available_preview',{}).get('frames')==self.canvas.duration:
                composed=selected.get('preview')
        self.direct_playback=bool(composed and Path(composed).is_file())
        self.playback_frames=int(self.canvas.media.get('frames',self.canvas.duration)) if self.direct_playback else self.canvas.duration
        self.audio_player.setVideoOutput(self.video_widget if self.direct_playback else None)
        path=self.canvas.audio_path
        source=composed if self.direct_playback else path
        self.audio_player.setSource(QUrl.fromLocalFile(str(source)) if source else QUrl())
        self.music_label.setText('音量' if source else '声音待生成')
        self.music_label.setToolTip('原作音乐与音效 · 按游戏帧同步' if path else '当前片段尚未配入音轨')
        self.chapters.clear()
        chapters=self.canvas.media.get('chapters',[dict(stage=1,start_frame=0)])
        for chapter in chapters:self.chapters.addItem(f'第 {chapter["stage"]} 面',int(chapter['start_frame']))
    def video_frame_ready(self,frame):
        if self.direct_playback and frame.isValid():self.stack.setCurrentWidget(self.video_widget)
    def audio_error(self,error,message):
        self.music_label.setText('音频不可用');self.music_label.setToolTip(message)
    def toggle(self):
        if not self.playing and self.canvas.frame>=self.playback_frames-1:self.seek(0)
        self.playing=not self.playing;self.first=self.canvas.frame;self.origin=time.perf_counter()
        if self.playing and (self.direct_playback or self.canvas.audio_path):
            self.audio_player.setPosition(round(self.first*1000/60));self.audio_player.play()
        else:self.audio_player.pause()
    def seek(self,value):
        if self.direct_playback:self.canvas.frame=int(value)
        else:self.canvas.set_frame(value)
        self.slider.setValue(int(value));self.first=value;self.origin=time.perf_counter()
        if self.direct_playback or self.canvas.audio_path:self.audio_player.setPosition(round(value*1000/60))
    def tick(self):
        now=time.perf_counter()
        if self.presentation is not None and now-self.last_status_poll>2:
            self.last_status_poll=now
            try:
                settings=read_shared_json(self.presentation)
                job=read_shared_json(settings['pipeline_status']);phase=job.get('phase')
                labels={'rendering_original_game_at_native_frame_resolution':'正在生成原作完整画面',
                        'composing_1080p60':'完整录像可播放 · 正在导出成片',
                        'completed':'完整成片已就绪 · 点击刷新',
                        'failed':'素材任务停止 · 请查看状态记录',
                        'source_recording_incomplete':'素材采集已停止 · 已保留样片',
                        'recording_did_not_clear':'本次素材未通关 · 已保留样片'}
                if phase in ('collecting_real_neural_recording','waiting_for_real_neural_recording'):
                    state=read_shared_json(Path(settings['collecting_recording'])/'status.json')
                    game=state.get('game',{});stage=game.get('stage')
                    label=f'完整素材采集 · 第{stage}面' if stage else '完整素材准备中'
                else:label=labels.get(phase,'正在准备完整素材')
                stage_job=settings.get('stage_publication_status')
                if stage_job:
                    progress=read_shared_json(stage_job)
                    if progress.get('phase') in ('rendering','composing','publishing'):
                        label+=f' · 第{progress.get("stage")}面正在生成'
                    elif progress.get('phase')=='parallel_preparing':
                        active=[]
                        for job in progress.get('parallel_jobs',[]):
                            try:phase=read_shared_json(job['status'])['phase']
                            except (OSError,ValueError,KeyError):phase='preparing'
                            if phase!='completed':active.append(str(job['stage']))
                        if active:label+=' · 第'+'/'.join(active)+'面并行生成'
                seconds=self.playback_frames//60
                available=f'可播放 {seconds//60:02d}:{seconds%60:02d}'
                if settings.get('available_preview',{}).get('frames',0)>self.playback_frames:
                    available+=' · 新片段已就绪，点击刷新'
                elif self.direct_playback and settings.get('preview') and Path(settings['preview'])!=Path(self.audio_player.source().toLocalFile()):
                    available+=' · 视频已更新，点击刷新'
                self.status_label.setText(available+'  |  '+label)
            except (OSError,ValueError,KeyError):pass
        if self.playing:
            if self.direct_playback:
                self.canvas.frame=min(self.playback_frames-1,round(self.audio_player.position()*60/1000))
                self.slider.setValue(self.canvas.frame)
                if self.audio_player.mediaStatus()==QMediaPlayer.EndOfMedia:self.playing=False
                return
            frame=self.first+int((time.perf_counter()-self.origin)*60)
            if frame>=self.canvas.duration:
                self.playing=False;frame=self.canvas.duration-1;self.audio_player.pause()
            self.canvas.set_frame(frame);self.slider.setValue(frame)
            if self.canvas.audio_path and now-self.last_audio_sync>.5:
                self.last_audio_sync=now
                target=round(frame*1000/60)
                if abs(self.audio_player.position()-target)>150:self.audio_player.setPosition(target)
    def clean(self):
        visible=self.controls.isVisible();self.controls.setVisible(not visible)
        self.showFullScreen() if visible else self.showNormal()
    def keyPressEvent(self,event):
        if event.key()==Qt.Key_F11:self.clean()
        elif event.key()==Qt.Key_Space:self.toggle()
        elif event.key()==Qt.Key_Escape and not self.controls.isVisible():self.clean()
        else:super().keyPressEvent(event)
    def save_frame(self):
        path,_=QFileDialog.getSaveFileName(self,'保存画面',str(self.canvas.recording/'presentation.png'),'PNG (*.png)')
        if path:
            if self.direct_playback:self.video_widget.videoSink().videoFrame().toImage().save(path)
            else:self.canvas.image().save(path)


def main():
    import argparse,sys,os
    p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--recording',type=Path);source.add_argument('--presentation',type=Path)
    p.add_argument('--screenshot',type=Path);p.add_argument('--frame',type=int,default=0)
    p.add_argument('--export',type=Path);p.add_argument('--frames',type=int,default=0)
    p.add_argument('--media',type=Path);p.add_argument('--progress',type=Path)
    p.add_argument('--ffmpeg',default='ffmpeg')
    args=p.parse_args()
    if args.presentation is not None:args.recording=Path(read_shared_json(args.presentation)['recording'])
    if args.screenshot or args.export:os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    app=QApplication(sys.argv)
    # The offscreen Qt platform does not enumerate Windows fonts itself.
    for font in ['msyh.ttc','msyhbd.ttc','segoeui.ttf','seguisym.ttf']:
        path=Path('C:/Windows/Fonts')/font
        if path.exists():QFontDatabase.addApplicationFont(str(path))
    if args.screenshot or args.export:
        canvas=DemoCanvas(args.recording,args.media);canvas.set_frame(args.frame)
        if args.screenshot:canvas.image().save(str(args.screenshot))
        if args.export:
            if canvas.video is None:raise RuntimeError('Original game frames have not been rendered yet')
            count=min(args.frames or canvas.duration,canvas.duration)
            composed=canvas.media.get('presentation_video')
            if composed and Path(composed).is_file() and count==canvas.duration:
                import shutil,uuid
                if Path(composed).resolve()!=args.export.resolve():
                    pending=args.export.with_name(args.export.name+'.'+uuid.uuid4().hex+'.publishing')
                    try:
                        # A completed recording on the same volume needs no second video allocation.
                        try:os.link(composed,pending)
                        except OSError:shutil.copy2(composed,pending)
                        pending.replace(args.export)
                    finally:pending.unlink(missing_ok=True)
                print(json.dumps(dict(path=str(args.export),frames=count,fps=60,audio=True,reused_completed_video=True)))
                return
            cmd=[args.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','1920x1080','-framerate','60','-i','pipe:0']
            if canvas.audio_path:
                cmd+=['-i',str(canvas.audio_path),'-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k']
            else:cmd+=['-an']
            cmd+=['-c:v','h264_nvenc','-preset','p5','-cq','18','-pix_fmt','yuv420p',
                '-t',str(count/60),'-movflags','+faststart',str(args.export)]
            child=subprocess.Popen(cmd,stdin=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                for frame in range(count):
                    canvas.set_frame(frame);im=canvas.image();child.stdin.write(bytes(im.constBits()))
                    if args.progress and frame%600==0:
                        from .thrl_durable_records import write_json
                        write_json(args.progress,dict(phase='composing',frames=frame+1,total_frames=count))
            finally:child.stdin.close()
            if child.wait()!=0:raise RuntimeError('Video encoding failed')
            if args.progress:
                from .thrl_durable_records import write_json
                write_json(args.progress,dict(phase='completed',frames=count,total_frames=count))
            print(json.dumps(dict(path=str(args.export),frames=count,fps=60,seconds=count/60,audio=bool(canvas.audio_path))))
    else:
        window=DemoWindow(args.recording,args.presentation);window.show();app.exec()
