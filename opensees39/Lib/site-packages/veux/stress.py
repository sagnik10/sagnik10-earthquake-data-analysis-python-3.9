import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri

from .elements import (
    quads_to_4tris,
    quads_to_8tris_9n,
    quads_to_8tris_8n,
    bricks_to_24tris,
    bricks_to_48tris,
    tetra4n_to_4tris,
    tetra10n_to_16tris,
    tris6n_to_4tris
)


class EleClassTag:
    """ELE_TAG constants defined in SRC/classTags.h"""
    truss = 12
    trussSection = 13
    CorotTruss = 14
    ZeroLength = 19
    ZeroLengthSection = 20
    CoupledZeroLength = 26
    ElasticBeam2d = 3
    ElasticBeam3d = 5
    DispBeamColumn2d = 62
    DispBeamColumn3d = 64
    ForceBeamColumn2d = 73
    ForceBeamColumn3d = 74
    TimoshenkoBeamColumn2d = 63
    TimoshenkoBeamColumn3d = 631
    ElasticTimoshenkoBeam2d = 145
    ElasticTimoshenkoBeam3d = 146
    tri3n = 33
    tri6n = 209
    quad4n = 31
    quad4n3d = 32
    quad9n = 207
    quad8n = 208
    SSPquad = 119
    EnhancedQuad = 59
    brick20n = 49
    brick8n = 56
    SSPbrick = 121
    FourNodeTetrahedron = 179
    TenNodeTetrahedron = 256
    TenNodeTetrahedronSK = 1790
    ASDShellQ4 = 203
    ASDShellT3 = 204
    ShellMITC4 = 53
    ShellMITC9 = 54
    ShellDKGQ = 156
    ShellNLDKGQ = 157
    ShellDKGT = 167
    ShellNLDKGT = 168        
    Joint2D = 82
    Joint3D = 83
    MVLEM = 162
    SFI_MVLEM = 163
    MVLEM_3D = 212
    SFI_MVLEM_3D = 213
    TwoNodeLink = 86
    Pipe = 269
    CurvedPipe = 270


class LoadTag:
    """LOAD_TAG constants defined in SRC/classTags.h"""
    Beam2dUniformLoad = 3
    Beam2dUniformLoad_ndata = 2

    Beam2dPointLoad = 4
    Beam2dPointLoad_ndata = 3

    Beam3dUniformLoad = 5
    Beam3dUniformLoad_ndata = 3

    Beam3dPointLoad = 6
    Beam3dPointLoad_ndata = 4

    BrickSelfWeight = 7
    Beam2dTempLoad = 8
    SurfaceLoader = 9
    SelfWeight = 10
    Beam2dThermalAction = 11

    Beam2dPartialUniformLoad = 12
    Beam2dPartialUniformLoad_ndata = 6

    Beam3dPartialUniformLoad = 121
    Beam3dPartialUniformLoad_ndata = 5

    Beam3dThermalAction = 13
    ShellThermalAction = 14
    NodalThermalAction = 15
    ThermalActionWrapper = 16
    LysmerVelocityLoader = 17


def _stress_2d_ele_tags_only(model, ele_tags):

    Stress2dEleClasstags = {EleClassTag.tri3n,
                            EleClassTag.tri6n,
                            EleClassTag.quad4n,
                            EleClassTag.quad8n,
                            EleClassTag.quad9n}

    ele_classtags = model.getEleClassTags()

    idx = (i for i, e in enumerate(ele_classtags) if e in Stress2dEleClasstags)

    ele_tags_2d_tris_quads_only = [ele_tags[i] for i in idx]

    return ele_tags_2d_tris_quads_only


def _sig_out_per_node(model, how_many='all'):
    """Return a 2d numpy array of stress components per OpenSees node.

    Three first stress components (sxx, syy, sxy) are calculated and
    extracted from OpenSees, while the rest svm (Huber-Mieses-Hencky),
    two principal stresses (s1, s2) and directional angle are calculated
    as postprocessed quantities.

    Args:
        how_many (str): supported options are: 'all' - all components,
            'sxx', 'syy', 'sxy', 'svm' (or 'vmis'), 's1', 's2', 'angle'.

    Returns:
        sig_out (ndarray): a 2d array of stress components per node with
            the following components: sxx, syy, sxy, svm, s1, s2, angle.
            Size (n_nodes x 7).

    Examples:
        sig_out = opsv.sig_out_per_node(model)

    Notes:
       s1, s2: principal stresses
       angle: angle of the principal stress s1
    """
    ele_tags = model.getEleTags()
    node_tags = model.getNodeTags()
    n_nodes = len(node_tags)

    ele_classtag = model.getEleClassTags(ele_tags[0])[0]

    # initialize helper arrays
    sig_out = np.zeros((n_nodes, 7))

    nodes_tag_count = np.zeros((n_nodes, 2), dtype=int)
    nodes_tag_count[:, 0] = node_tags

    nen = np.shape(model.eleNodes(ele_tags[0]))[0]

    for i, ele_tag in enumerate(ele_tags):
        ele_node_tags = model.eleNodes(ele_tag)

        tmp_list = [0]*nen
        for j, ele_node_tag in enumerate(ele_node_tags):
            tmp_list[j] = node_tags.index(ele_node_tag)

        nodes_tag_count[tmp_list, 1] += 1

        if ele_classtag == EleClassTag.SSPquad:
            sig_ip_el = model.eleResponse(ele_tag, 'stress')
            # sigM_nd = sigM_ip
            sigM_np = np.tile(sig_ip_el, (nen, 1))

        else:
            sig_nd_el = model.eleResponse(ele_tag, 'stressAtNodes')
            sigM_nd = np.reshape(sig_nd_el, (-1, 3))

        # sxx yy xy components
        sig_out[tmp_list, 0] += sigM_nd[:nen, 0]
        sig_out[tmp_list, 1] += sigM_nd[:nen, 1]
        sig_out[tmp_list, 2] += sigM_nd[:nen, 2]

    indxs, = np.where(nodes_tag_count[:, 1] > 1)

    # n_indxs < n_nodes: e.g. 21<25 (bous), 2<6 (2el) etc.
    n_indxs = np.shape(indxs)[0]

    # divide summed stresses by the number of common nodes
    sig_out[indxs, :] = sig_out[indxs, :] / nodes_tag_count[indxs, 1].reshape(n_indxs, 1)

    if how_many == 'all' or how_many == 'svm' or how_many == 'vmis':
        # warning reshape from (pts,ncomp) to (ncomp,pts)
        sig_out[:, 3] = vm_stress(sig_out[:, :3].T)

    if how_many == 'all' or how_many == 's1' or how_many == 's2' or how_many == 'angle':
        sig_out[:, 4:7] = principal_stress(sig_out[:, :3].T).T

    return sig_out


def node_average(model, response, ndm=2, keys=None):
    """
    Return a dictionary of stress components per node.
    For 2D: sxx, syy, sxy
    For 3D: sxx, syy, szz, sxy, syz, sxz
    Summed values from each element's node contribution are averaged at the end.

    Args:
        model: an OpenSees model-like object with the following interface:
               - getEleTags()
               - getEleClassTags(eleTag)
               - getNodeTags()
               - eleNodes(eleTag)
               - eleResponse(eleTag, str)
        response (str): a label if needed for e.g. 'stress', 'stressAtNodes'
        ndm (int): number of dimensions (2 or 3)

    Returns:
        dict[node_tag] = {
            "xx": float,
            "yy": float,
            ...
        }
    """

    # Helper: pick which elements are 2D vs 3D, if needed
    # For example, you might have separate function(s) or logic:
    #    ele_tags = _stress_2d_ele_tags_only(model, model.getEleTags())
    # or for 3D:
    #    ele_tags = _stress_3d_ele_tags_only(model, model.getEleTags())
    #
    # Below, we simply re-use the user's helper for 2D, but you'll have to define
    # your own for 3D or unify them if your model can contain both:
    ele_tags = model.getEleTags()

    # For a typical 2D element with 3 stress components: sxx, syy, sxy
    # For a typical 3D element with 6 stress components: sxx, syy, szz, sxy, syz, sxz
    # You may need to adapt this depending on your element type(s).
    if ndm == 2:
        keys = "sxx", "syy", "sxy"
    else:
        keys = "sxx", "syy", "szz", "sxy", "syz", "sxz"


    node_tags = model.getNodeTags()

    # Initialize a dictionary to hold stress sums and a count of how many
    # times each node is encountered (to do averaging).
    sig_dict = {node: {"_count": 0, **{key: 0.0 for key in keys}} for node in node_tags}

    nrc = len(keys)

    # We assume each element’s response yields a flat array 
    # with length = n_nodes_in_element * nrc.
    # Then we reshape it to (nen, nrc).
    for elem in ele_tags:
        ele_node_tags = model.eleNodes(elem)
        nen = len(ele_node_tags)

        sig_per_node = np.reshape(model.eleResponse(elem, response), (nen, nrc))

        # Accumulate into sig_dict
        for i, node in enumerate(ele_node_tags):
            for j, key in enumerate(keys):
                sig_dict[node][key] += sig_per_node[i][j]

            sig_dict[node]["_count"] += 1

    # Now do averaging (if a node belongs to multiple elements).
    for node in node_tags:
        c = sig_dict[node]["_count"] or 1.0

        for key in keys:
            sig_dict[node][key] /= c


    # Clean up
    for node in node_tags:
        sig_dict[node].pop("_count", None)

    return sig_dict

def _sig_per_node(model):

    ele_tags_all = model.getEleTags()
    ele_tags = _stress_2d_ele_tags_only(model, ele_tags_all)

    ele_classtag = model.getEleClassTags(ele_tags[0])[0]

    node_tags = model.getNodeTags()
    n_nodes = len(node_tags)

    # initialize helper arrays
    sig_out = np.zeros((n_nodes, 4))

    nodes_tag_count = np.zeros((n_nodes, 2), dtype=int)
    nodes_tag_count[:, 0] = node_tags

    nen = np.shape(model.eleNodes(ele_tags[0]))[0]

    for i, ele_tag in enumerate(ele_tags):
        ele_node_tags = model.eleNodes(ele_tag)

        tmp_list = [0]*nen
        for j, ele_node_tag in enumerate(ele_node_tags):
            tmp_list[j] = node_tags.index(ele_node_tag)

        nodes_tag_count[tmp_list, 1] += 1

        if ele_classtag == EleClassTag.SSPquad:
            sig_ip_el = model.eleResponse(ele_tag, 'stress')
            # sigM_nd = sigM_ip
            sigM_np = np.tile(sig_ip_el, (nen, 1))

        else:
            sig_nd_el = model.eleResponse(ele_tag, 'stressAtNodes')
            sigM_nd = np.reshape(sig_nd_el, (-1, 3))

        # sxx yy xy components
        sig_out[tmp_list, 0] += sigM_nd[:nen, 0] # xx
        sig_out[tmp_list, 1] += sigM_nd[:nen, 1] # yy
        sig_out[tmp_list, 2] += sigM_nd[:nen, 2] # xy

    indxs, = np.where(nodes_tag_count[:, 1] > 1)

    # n_indxs < n_nodes: e.g. 21<25 (bous), 2<6 (2el) etc.
    n_indxs = np.shape(indxs)[0]

    # divide summed stresses by the number of common nodes
    sig_out[indxs, :] = \
        sig_out[indxs, :]/nodes_tag_count[indxs, 1].reshape(n_indxs, 1)
    
    return sig_out

def _sig_component_per_node(model, stress_str):
    """Return a 2d numpy array of stress components per OpenSees node.

    Three first stress components (sxx, syy, sxy) are calculated and
    extracted from OpenSees, while the rest svm (Huber-Mieses-Hencky),
    two principal stresses (s1, s2) and directional angle are calculated
    as postprocessed quantities.

    Args:
        how_many (str): supported options are: 'all' - all components,
            'sxx', 'syy', 'sxy', 'svm' (or 'vmis'), 's1', 's2', 'angle'.

    Returns:
        sig_out (ndarray): a 2d array of stress components per node with
            the following components: sxx, syy, sxy, svm, s1, s2, angle.
            Size (n_nodes x 7).


    Notes:
       s1, s2: principal stresses
       angle: angle of the principal stress s1
    """
    sig_out = _sig_per_node(model) 

    if stress_str == 'sxx':
        sig_out_vec = sig_out[:, 0]
    elif stress_str == 'syy':
        sig_out_vec = sig_out[:, 1]
    elif stress_str == 'sxy':
        sig_out_vec = sig_out[:, 2]
    elif stress_str == 'svm' or stress_str == 'vmis':
        # warning reshape from (pts,ncomp) to (ncomp,pts)
        sig_out_vec = vm_stress(np.transpose(sig_out[:, :3]))
    elif (stress_str == 's1' or stress_str == 's2' or stress_str == 'angle'):
        princ_sig_out = principal_stress(np.transpose(sig_out[:, :3]))
        if stress_str == 's1':
            # sig_out_vec = np.transpose(princ_sig_out)[:, 0]
            sig_out_vec = princ_sig_out[0, :]
        elif stress_str == 's2':
            sig_out_vec = princ_sig_out[1, :]
        elif stress_str == 'angle':
            sig_out_vec = princ_sig_out[2, :]

    return sig_out_vec


def principal_stress(sig):
    """Return a tuple (s1, s2, angle): principal stresses (plane stress) and angle
    Args:
        sig (ndarray): input array of stresses at nodes: sxx, syy, sxy (tau)

    Returns:
        out (ndarray): 1st row is first principal stress s1, 2nd row is second
           principal stress s2, 3rd row is the angle of s1
    """
    sx, sy, tau = sig[0], sig[1], sig[2]

    ds = (sx-sy)/2
    R = np.sqrt(ds**2 + tau**2)

    s1 = (sx+sy)/2. + R
    s2 = (sx+sy)/2. - R
    angle = np.arctan2(tau, ds)/2

    out = np.vstack((s1, s2, angle))

    return out


def vm_stress(sig):
    n_sig_comp, n_pts = np.shape(sig)
    if n_sig_comp > 3:
        x, y, z, xy, xz, yz = sig
    else:
        x, y, xy = sig
        z, xz, yz = 0., 0., 0.

    _a = 0.5*((x-y)**2 + (y-z)**2 + (z-x)**2 + 6.*(xy**2 + xz**2 + yz**2))
    return np.sqrt(_a)



def plot_mesh_2d(nds_crd, eles_conn, lw=0.4, ec='k'):
    """
    Plot 2d mesh (quads or triangles) outline.
    """
    nen = np.shape(eles_conn)[1]
    if nen == 3 or nen == 4:
        for ele_conn in eles_conn:
            x = nds_crd[ele_conn, 0]
            y = nds_crd[ele_conn, 1]
            plt.fill(x, y, edgecolor=ec, lw=lw, fill=False)

    elif nen == 6:
        for ele_conn in eles_conn:
            x = nds_crd[[ele_conn[0], ele_conn[3], ele_conn[1], ele_conn[4],
                         ele_conn[2], ele_conn[5]], 0]
            y = nds_crd[[ele_conn[0], ele_conn[3], ele_conn[1], ele_conn[4],
                         ele_conn[2], ele_conn[5]], 1]
            plt.fill(x, y, edgecolor=ec, lw=lw, fill=False)

    elif nen == 9:
        for ele_conn in eles_conn:
            x = nds_crd[[ele_conn[0], ele_conn[4], ele_conn[1], ele_conn[5],
                         ele_conn[2], ele_conn[6], ele_conn[3], ele_conn[7]],
                        0]
            y = nds_crd[[ele_conn[0], ele_conn[4], ele_conn[1], ele_conn[5],
                         ele_conn[2], ele_conn[6], ele_conn[3], ele_conn[7]],
                        1]
            plt.fill(x, y, edgecolor=ec, lw=lw, fill=False)

    elif nen == 8:
        for ele_conn in eles_conn:
            x = nds_crd[[ele_conn[0], ele_conn[4], ele_conn[1], ele_conn[5],
                         ele_conn[2], ele_conn[6], ele_conn[3], ele_conn[7]],
                        0]
            y = nds_crd[[ele_conn[0], ele_conn[4], ele_conn[1], ele_conn[5],
                         ele_conn[2], ele_conn[6], ele_conn[3], ele_conn[7]],
                        1]
            plt.fill(x, y, edgecolor=ec, lw=lw, fill=False)


def plot_stress_2d(model, nds_val, mesh_outline=1, cmap='turbo', levels=50):
    """
    Plot stress distribution of a 2d elements of a 2d model.

    Args:
        nds_val (ndarray): the values of a stress component, which can
            be extracted from sig_out array (see sig_out_per_node
            function)

        mesh_outline (int): 1 - mesh is plotted, 0 - no mesh plotted.

        cmap (str): Matplotlib color map (default is 'turbo')

    Usage:
        See demo_quads_4x4.py example.
    """

    node_tags, ele_tags_all = model.getNodeTags(), model.getEleTags()

    ele_tags = _stress_2d_ele_tags_only(model, ele_tags_all)

    n_nodes, n_eles = len(node_tags), len(ele_tags)

    ele_classtag = model.getEleClassTags(ele_tags[0])[0]

    if (ele_classtag == EleClassTag.tri3n):
        nen = 3

    elif (ele_classtag == EleClassTag.tri6n):
        nen = 6

    elif (ele_classtag == EleClassTag.quad4n):
        nen = 4

    elif (ele_classtag == EleClassTag.quad8n):
        nen = 8

    elif (ele_classtag == EleClassTag.quad9n):
        nen = 9

    # nen = np.shape(model.eleNodes(ele_tags[0]))[0]

    # idiom coordinates as ordered in node_tags
    # use node_tags.index(tag) for correspondence
    nds_crd = np.zeros((n_nodes, 2))
    for i, node_tag in enumerate(node_tags):
        nds_crd[i] = model.nodeCoord(node_tag)

    # from utils / sig_out_per_node
    # fixme: if this can be simplified
    # index (starts from 0) to node_tag correspondence
    # (a) data in np.array of integers
    # nodes_tag_count = np.zeros((n_nodes, 2), dtype=int)
    # nodes_tag_count[:, 0] = node_tags
    #
    # correspondence indx and node_tag is in node_tags.index
    # after testing remove the above

    eles_conn = np.zeros((n_eles, nen), dtype=int)

    for i, ele_tag in enumerate(ele_tags):
        ele_node_tags = model.eleNodes(ele_tag)

        for j, ele_node_tag in enumerate(ele_node_tags):
            eles_conn[i, j] = node_tags.index(ele_node_tag)

    if ele_classtag == EleClassTag.tri3n:
        tris_conn = eles_conn
        nds_crd_all = nds_crd
        nds_val_all = nds_val

    elif ele_classtag == EleClassTag.quad4n:
        tris_conn, nds_c_crd, nds_c_val = \
            quads_to_4tris(eles_conn, nds_crd, nds_val)

        nds_crd_all = np.vstack((nds_crd, nds_c_crd))
        nds_val_all = np.hstack((nds_val, nds_c_val))

    elif ele_classtag == EleClassTag.tri6n:
        tris_conn = tris6n_to_4tris(eles_conn)
        nds_crd_all = nds_crd
        nds_val_all = nds_val

    elif ele_classtag == EleClassTag.quad8n:
        tris_conn, nds_c_crd, nds_c_val = quads_to_8tris_8n(eles_conn,
                                                            nds_crd, nds_val)

        nds_crd_all = np.vstack((nds_crd, nds_c_crd))
        nds_val_all = np.hstack((nds_val, nds_c_val))

    elif ele_classtag == EleClassTag.quad9n:
        tris_conn = quads_to_8tris_9n(eles_conn)
        nds_crd_all = nds_crd
        nds_val_all = nds_val



    # 1. plot contour maps
    triangulation = tri.Triangulation(nds_crd_all[:, 0],
                                      nds_crd_all[:, 1],
                                      tris_conn)

    plt.tricontourf(triangulation, nds_val_all, levels=levels, cmap=cmap)

    # 2. plot original mesh (quad) without subdivision into triangles
    if mesh_outline:
        plot_mesh_2d(nds_crd, eles_conn)

    plt.colorbar()
    plt.axis('equal')
    plt.grid(False)


def plot_stress(stress_str, mesh_outline=1, cmap='turbo', levels=50):
    """Plot stress distribution of the model.

    Args:
        stress_str (string): stress component string. Available options are:
            'sxx', 'syy', 'sxy', 'vmis', 's1', 's2', 'alpha'

        mesh_outline (int): 1 - mesh is plotted, 0 - no mesh plotted.

        cmap (str): Matplotlib color map (default is 'turbo')

        levels (int): number and positions of the contour lines / regions.

    Usage:
        ::

            opsv.plot_stress('vmis')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.show()

    See also:

    :ref:`opsvis_sig_out_per_node`
    """

    ndim = model.getNDM()[0]

    if ndim == 2:
        _plot_stress_2d(stress_str, mesh_outline, cmap, levels)

    # not implemented yet
    # elif ndim == 3:
    #     _plot_stress_3d(stress_str, mesh_outline, cmap, levels)

    else:
        print(f'\nWarning! ndim: {ndim} not implemented yet.')

    # plt.show()  # call this from main py file for more control


def _plot_stress_2d(stress_str, mesh_outline, cmap, levels):
    """See documentation for plot_stress command"""

    # node_tags = model.getNodeTags()
    # ele_tags = model.getEleTags()
    # n_nodes = len(node_tags)

    # second version - better - possible different types
    # of elements (mix of quad and tri)
    # for ele_tag in ele_tags:
    #     nen = np.shape(model.eleNodes(ele_tag))[0]

    # avoid calculating and storing all stress components
    # sig_out = sig_out_per_node(model, stress_str)
    # switcher = {'sxx': 0,
    #             'syy': 1,
    #             'sxy': 2,
    #             'svm': 3,
    #             'vmis': 3,
    #             's1': 4,
    #             's2': 5,
    #             'angle': 6}

    # nds_val = sig_out[:, switcher[stress_str]]

    nds_val = _sig_component_per_node(model, stress_str)
    plot_stress_2d(nds_val, mesh_outline, cmap, levels)
