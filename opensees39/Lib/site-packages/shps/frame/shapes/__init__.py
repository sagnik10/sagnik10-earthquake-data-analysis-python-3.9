from .._patch import _patch as patch, SectionGeometry

def _WideFlange(aisc_data, mesh_data, material)->"SectionGeometry":

    if isinstance(aisc_data, dict):
        d  = aisc_data['d']
        bf = aisc_data['bf']
        tf = aisc_data['tf']
        tw = aisc_data['tw']

    if isinstance(mesh_data, dict):
        nft = mesh_data['nft']
        nwl = mesh_data['nwl']
        nfl = mesh_data.get('nfl', 1, ) #mesh_data['nft'])
        nwt = mesh_data.get('nwt', 1, ) #mesh_data['nwl'])

        if ndm is None:
            ndm = mesh_data.get("ndm", 3)

        int_typ = mesh_data.get("IntTyp", None)
        flg_opt = mesh_data.get('FlgOpt', True)
#       web_opt = mesh_data.get('WebOpt', False)

    else:
        assert isinstance(mesh_data, tuple)
        nft, nwl = mesh_data
        nfl, nwt = 1, 1
        int_typ  = None
        flg_opt  = True

    yoff = ( d - tf) / 2
    zoff = (bf + tw) / 4

    dw = d - 2 * tf
    bi = bf - tw

    if ndm == 2:
        GJ =  1.0

    else:
        J = aisc_data.get("J")

        # return SectionGeometry(name=tag, GJ=GJ, areas=[
        #     patch.rect(vertices=[[-],[]], material=material)
        #     # patch.Fiber([x,y], A, material) for x,y,A in zip(xfib, yfib, wfib)
        # ])

    if flg_opt:
        return SectionGeometry(shapes=[
            patch.rect(corners=[[-bf/2, yoff-tf/2],[bf/2,  yoff+tf/2]], material=material, divs=(nfl, nft), rule=int_typ),
            patch.rect(corners=[[-tw/2,-yoff+tf/2],[tw/2,  yoff-tf/2]], material=material, divs=(nwt, nwl), rule=int_typ),
            patch.rect(corners=[[-bf/2,-yoff-tf/2],[bf/2, -yoff+tf/2]], material=material, divs=(nfl, nft), rule=int_typ),
        ])

    else:
        return SectionGeometry(shapes=[
            patch.rect(corners=[[-zoff-bi/4, yoff-tf/2],[-zoff+bi/4,  yoff+tf/2]], material=material, divs=(nfl, nft), rule=int_typ),
            patch.rect(corners=[[ zoff-bi/4, yoff-tf/2],[ zoff+bi/4,  yoff+tf/2]], material=material, divs=(nfl, nft), rule=int_typ),
            patch.rect(corners=[[     -tw/2,-yoff-tf/2],[      tw/2,  yoff+tf/2]], material=material, divs=(nwt, nwl), rule=int_typ),
            patch.rect(corners=[[-zoff-bi/4,-yoff-tf/2],[-zoff+bi/4, -yoff+tf/2]], material=material, divs=(nfl, nft), rule=int_typ),
            patch.rect(corners=[[ zoff-bi/4,-yoff-tf/2],[ zoff+bi/4, -yoff+tf/2]], material=material, divs=(nfl, nft), rule=int_typ),
        ])


def from_shape(type, identifier: str, material=None, mesh=None, units=None, ndm=None, tag=None, **kwds):
    if identifier == "WF":
        # TODO
        return _WideFlange

    else:
        return from_aisc(type, identifier, material, mesh, units, ndm, tag, **kwds)

