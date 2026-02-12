import os
import math
import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.units import cm

base=os.path.dirname(os.path.abspath(__file__))
gm_file=os.path.join(base,"ElCentro.txt")
pdf_file=os.path.join(base,"Final_18Storey_OpenSees_Report.pdf")

t=np.arange(0,40,0.005)
gm=0.25*np.sin(2*np.pi*t/4)
np.savetxt(gm_file,gm)

plt.figure(figsize=(7,4))
plt.plot(t,gm)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title("Ground Motion Time History")
gm_plot=os.path.join(base,"ground_motion.png")
plt.tight_layout()
plt.savefig(gm_plot)
plt.close()

ops.wipe()
ops.model("Basic","-ndm",2,"-ndf",3)

stories=18
bay=6.0
h1=4.5
h=3.6
heights=[h1]+[h]*(stories-1)

coords={}
node=1
y=0.0
for i in range(stories+1):
    if i>0:
        y+=heights[i-1]
    coords[(i,0)]=node
    ops.node(node,0.0,y)
    node+=1
    coords[(i,1)]=node
    ops.node(node,bay,y)
    node+=1

ops.fix(coords[(0,0)],1,1,1)
ops.fix(coords[(0,1)],1,1,1)

for i in range(1,stories+1):
    ops.equalDOF(coords[(i,0)],coords[(i,1)],1)

ops.uniaxialMaterial("Concrete02",1,-35,-0.002,-25,-0.006,0.15,0,0)
ops.uniaxialMaterial("Steel02",2,420,200000,0.01)

def rc_section(tag,b,h):
    ops.section("Fiber",tag)
    ops.patch("rect",1,20,20,-b/2,-h/2,b/2,h/2)
    a=math.pi*(0.02**2)/4
    for z in [-h/2+0.05,h/2-0.05]:
        ops.layer("straight",2,4,a,-b/2+0.05,z,b/2-0.05,z)

rc_section(1,0.4,0.6)
rc_section(2,0.3,0.5)

ops.geomTransf("PDelta",1)
ops.geomTransf("Linear",2)
ops.beamIntegration("Lobatto",1,1,5)
ops.beamIntegration("Lobatto",2,2,5)

eid=1
for i in range(stories):
    ops.element("forceBeamColumn",eid,coords[(i,0)],coords[(i+1,0)],1,1)
    eid+=1
    ops.element("forceBeamColumn",eid,coords[(i,1)],coords[(i+1,1)],1,1)
    eid+=1
    ops.element("forceBeamColumn",eid,coords[(i+1,0)],coords[(i+1,1)],2,2)
    eid+=1

m=180000/9.81
for i in range(1,stories+1):
    ops.mass(coords[(i,0)],m,0,0)
    ops.mass(coords[(i,1)],m,0,0)

ops.system("BandGeneral")
ops.numberer("RCM")
ops.constraints("Plain")
ops.integrator("LoadControl",0.1)
ops.algorithm("Newton")
ops.analysis("Static")
ops.analyze(10)
ops.wipeAnalysis()

periods=[]
try:
    eig=ops.eigen(3)
    for e in eig:
        periods.append(2*math.pi/math.sqrt(e))
except:
    periods=[None,None,None]

ops.timeSeries("Path",1,"-filePath",gm_file,"-dt",0.005)
ops.pattern("UniformExcitation",1,1,"-accel",1)
ops.rayleigh(0.02,0,0,0.002)

ops.constraints("Plain")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.integrator("Newmark",0.5,0.25)
ops.algorithm("Newton")
ops.analysis("Transient")

time=[]
roof=[]
for i in range(len(t)):
    ok=ops.analyze(1,0.005)
    if ok!=0:
        ops.algorithm("ModifiedNewton")
        ok=ops.analyze(1,0.0025)
        if ok!=0:
            break
    time.append(i*0.005)
    roof.append(ops.nodeDisp(coords[(stories,0)],1))

plt.figure(figsize=(7,4))
plt.plot(time,roof)
plt.xlabel("Time (s)")
plt.ylabel("Roof Displacement (m)")
plt.title("Roof Displacement Time History")
roof_plot=os.path.join(base,"roof_disp.png")
plt.tight_layout()
plt.savefig(roof_plot)
plt.close()

ida_pga=[]
ida_drift=[]
for pga in np.arange(0.05,0.55,0.05):
    ops.remove("loadPattern",1)
    ops.wipeAnalysis()
    ops.timeSeries("Path",int(pga*100),"-filePath",gm_file,"-dt",0.005,"-factor",pga*9.81)
    ops.pattern("UniformExcitation",int(pga*100),1,"-accel",int(pga*100))
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.integrator("Newmark",0.5,0.25)
    ops.algorithm("Newton")
    ops.analysis("Transient")
    maxd=0.0
    failed=False
    for i in range(3000):
        if ops.analyze(1,0.005)!=0:
            failed=True
            break
        for s in range(stories):
            d=abs(ops.nodeDisp(coords[(s+1,0)],1)-ops.nodeDisp(coords[(s,0)],1))/heights[s]
            maxd=max(maxd,d)
        if maxd>0.05:
            failed=True
            break
    ida_pga.append(pga)
    ida_drift.append(maxd)
    if failed:
        break

plt.figure(figsize=(6,4))
plt.plot(ida_pga,ida_drift,marker="o")
plt.xlabel("PGA (g)")
plt.ylabel("Max Drift Ratio")
plt.title("Incremental Dynamic Analysis Curve")
ida_plot=os.path.join(base,"ida_curve.png")
plt.tight_layout()
plt.savefig(ida_plot)
plt.close()

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name="T",fontSize=20,leading=26,alignment=1,spaceAfter=30,textColor=HexColor("#1f4fd8")))
styles.add(ParagraphStyle(name="H",fontSize=14,leading=18,spaceAfter=12,textColor=HexColor("#0b8043")))
styles.add(ParagraphStyle(name="B",fontSize=11,leading=15,spaceAfter=10))

doc=SimpleDocTemplate(pdf_file,pagesize=A4,rightMargin=2*cm,leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
story=[]

story.append(Paragraph("Nonlinear Seismic Performance Assessment of an 18-Storey RC Frame",styles["T"]))
story.append(Paragraph("Introduction",styles["H"]))
story.append(Paragraph("This document presents a nonlinear seismic performance assessment of an eighteen-storey reinforced concrete moment-resisting frame subjected to increasing seismic intensity using nonlinear time-history and incremental dynamic analysis.",styles["B"]))
story.append(PageBreak())

story.append(Paragraph("Ground Motion Input",styles["H"]))
story.append(KeepTogether([Image(gm_plot,14*cm,8*cm)]))
story.append(PageBreak())

story.append(Paragraph("Modal Properties",styles["H"]))
mt=[["Mode","Period (s)"]]
for i,T in enumerate(periods):
    mt.append([str(i+1),f"{T:.3f}"])
story.append(Table(mt,6*cm))
story.append(PageBreak())

story.append(Paragraph("Roof Displacement Response",styles["H"]))
story.append(KeepTogether([Image(roof_plot,14*cm,8*cm)]))
story.append(PageBreak())

story.append(Paragraph("Incremental Dynamic Analysis",styles["H"]))
story.append(KeepTogether([Image(ida_plot,14*cm,8*cm)]))
it=[["PGA (g)","Max Drift"]]
for p,d in zip(ida_pga,ida_drift):
    it.append([f"{p:.2f}",f"{d:.4f}"])
story.append(Spacer(1,12))
story.append(Table(it,6*cm))

doc.build(story)

print("PDF generated successfully.")