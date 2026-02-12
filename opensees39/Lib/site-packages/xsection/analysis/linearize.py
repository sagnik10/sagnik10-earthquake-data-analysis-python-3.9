import numpy as np

def linearize(model):
    tangent = model.invoke("section", 1, [
                           "update  "+" ".join(["0"]*12)+";",
                           "tangent"
            ])

    n = int(np.sqrt(len(tangent)))
    Ks = np.array(tangent).reshape(n,n).T


#   plt.spy(Ks)
#   plt.imshow(Ks,interpolation='none',cmap='binary')
#   plt.show()
    # Ks = np.round(Ks, 4)
    # print(pd.DataFrame(Ks))

    cnn = Ks[:3,:3]
    cmm = Ks[3:6,3:6]
    cnm = Ks[:3,3:6]
    cnw = Ks[:3,6:9]
    cnv = Ks[:3,9:12]
    cmw = Ks[3:6,6:9]
    cmv = Ks[3:6,9:12]
    cww = Ks[6:9,6:9]
    cvv = Ks[9:12,9:12]
    # print(f"{cnm = }")
    # print(f"{cnw = }")

    tol=1e-14

    G, E = 1,1
    EA = cnn[0,0]
    A  = EA/E
    GA = G*A

    Qy = cnm[0,1] # int z
    Qz = cnm[2,0] # int y
    # Compute centroid
    cx, cy = float(Qz/GA), float(Qy/EA)
    cx, cy = map(lambda i: i if abs(i)>tol else 0.0, (cx, cy))


    Ivv = cvv[0,0]
    Isv = cmm[0,0] - Ivv

    s = f"""

  [nn]    Area               {A           :>10.4}
  [nm]    Centroid           {0.0         :>10.4}  {cx          :>10.4}, {cy          :>10.4}
  [nw|v]                     {cnw[0,0]/GA :>10.4}  {cnv[1,0]/GA :>10.4}, {cnv[2,0]/GA :>10.4}

  [mm]    Flexural moments   {cmm[0,0]/G  :>10.4}  {cmm[1,1]/E  :>10.4}, {cmm[2,2]/E  :>10.4}, {cmm[1,2]/E :>10.4}
  [mv|w]                     {cmv[0,0]/G  :>10.4}  {cmw[1,0]/E  :>10.4}, {cmw[2,0]/E  :>10.4}


  [ww]    Warping constant   {cww[0,0]/E  :>10.4}
          Torsion constant   {Isv/G       :>10.4}
  [vv]    Bishear            {Ivv/G       :>10.4}
    """
    print(s)