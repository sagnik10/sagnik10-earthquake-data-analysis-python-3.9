

"""
ExtGirdType=AASHTO-PCI-ASBI  
    TotalWidth=330.708661417323
    TotalDepth=70.8661417322835
    ExtGRadClip=22.8346456692913   
    GirdVert=No   
    TopSlabThk=8.85826771653543   
    BotSlabThk=7.8740157480315
    ExtGirdThk=10.5905511811024   
    TopSlabProf=Flat   
    LOLength=53.9370078740157   
    LOOuterThk=8.85826771653543   
    ROLength=53.9370078740157   
    ROOuterThk=8.85826771653543   
    FHoriz1=53.9370078740157   
    FHoriz2=47.244094488189   
    FHoriz3=0
    FVert1=4.92125984251969   
    FVert2=4.92125984251969   
    FVert3=0   
    CurbLoc="Program Determined"   
    CurbLeft=15   
    CurbRight=15   
    CurbMedian=0   
    CurbMWidth=0   
    RefPtOffX=0   
    RefPtOffY=0   
    DsgnTopDis=13.7795275590551   
    DsgnBotDis=7.8740157480315"""


from xsection.polygon import PolygonSection
import numpy as np

class AASHTO_PCI_ASBI(PolygonSection):
    def __init__(self,
                 width: float,
                 depth: float,
                 
                 thickness_top: float,
                 thickness_bot: float):
        self._width = width