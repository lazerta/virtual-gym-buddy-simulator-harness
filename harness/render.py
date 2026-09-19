from __future__ import annotations
import cv2, numpy as np
from pathlib import Path
from .profiles import SYNTHETIC_PERSONAL_REFERENCE, EXERCISES
from .scenarios import deterministic_scenario
from .simulator import simulate

def render_demo(path:str, frames=180):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'); out=cv2.VideoWriter(str(p),fourcc,30,(960,540))
    ex=EXERCISES[0]
    families=['clean','bystander_cross','occlusion','camera_bump','repeated_issue']
    for fi in range(frames):
        fam=families[min(len(families)-1,fi//36)]; s=deterministic_scenario(fam,900+fi); obs=simulate(SYNTHETIC_PERSONAL_REFERENCE,ex,s)
        img=np.full((540,960,3),28,np.uint8)
        for x in range(60,900,120): cv2.line(img,(x,80),(x,460),(65,65,65),5)
        cv2.rectangle(img,(260,390),(700,420),(75,75,75),-1)
        cx,cy=480,180
        head=(cx,cy); neck=(cx,cy+35); hip=(cx,cy+150)
        cv2.circle(img,head,26,(150,185,215),-1); cv2.line(img,neck,hip,(70,90,115),50)
        phase=np.sin(fi*.22)*.5+.5
        shL=(cx-45,cy+55); shR=(cx+45,cy+55); elL=(cx-95,int(cy+95-50*phase)); elR=(cx+95,int(cy+95-50*phase)); wrL=(cx-75,int(cy+45-95*phase)); wrR=(cx+75,int(cy+45-95*phase))
        for a,b in [(shL,elL),(elL,wrL),(shR,elR),(elR,wrR)]: cv2.line(img,a,b,(160,190,220),18)
        for w in [wrL,wrR]: cv2.rectangle(img,(w[0]-23,w[1]-8),(w[0]+23,w[1]+8),(25,25,25),-1)
        if fam=='bystander_cross':
            bx=int(120+(fi%36)*20); cv2.circle(img,(bx,200),24,(170,150,130),-1); cv2.line(img,(bx,224),(bx,410),(95,80,70),34)
        if fam=='occlusion': cv2.rectangle(img,(350,165),(470,370),(95,95,95),-1)
        if fam=='camera_bump': img=np.roll(img,20,axis=1)
        cv2.putText(img,f"{ex.id} | {fam}",(25,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(235,235,235),2)
        cv2.putText(img,f"fill={obs.frame_fill:.2f} track={obs.tracking_quality:.2f} people={obs.detected_people}",(25,70),cv2.FONT_HERSHEY_SIMPLEX,.65,(210,210,210),2)
        out.write(img)
    out.release(); return str(p)
