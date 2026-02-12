import numpy as np 


def quad8n_val_at_center(vals):
    """
    Calculate values at the center of 8-node quad element.
    """

    val_c1 = -np.mean(vals[[0, 1, 2, 3]], axis=0) + 2*np.mean(vals[[4, 5, 6, 7]], axis=0)  # noqa: E501
    # val_c1 = -np.sum(vals[[0, 1, 2, 3]], axis=0)/4 + np.sum(vals[[4, 5, 6, 7]], axis=0)/2  # noqa: E501
    # val_c2 = -np.sum(vals[[0, 1, 2, 3]], axis=0)/4 + np.sum(vals[[4, 5, 6, 7]], axis=0)/2  # noqa: E501
    # val_c3 = np.mean(vals[[4, 5, 6, 7]], axis=0)
    # val_c4 = np.mean(vals[[0, 1, 2, 3]], axis=0)
    # val_c5 = np.mean(vals, axis=0)
    # return val_c1, val_c2, val_c3, val_c4, val_c5

    return val_c1


def quads_to_4tris(quads_conn, nds_crd, nds_val):
    """
    Get triangles connectivity

    Notes:
        Triangles connectivity array is based on
        quadrilaterals connectivity.
        Each quad is split into four triangles.
        New nodes are created at the quad centroid.

    See also:
        function: quads_to_8tris_9n, quads_to_8tris_8n
    """
    n_quads, _ = quads_conn.shape
    n_nds, _ = nds_crd.shape

    tris_conn = np.zeros((4*n_quads, 3), dtype=int)

    for i, quad_conn in enumerate(quads_conn):
        j = 4*i
        n0, n1, n2, n3 = quad_conn

        # triangles connectivity
        tris_conn[j  ] = np.array([n0, n1, n_nds+i])
        tris_conn[j+1] = np.array([n1, n2, n_nds+i])
        tris_conn[j+2] = np.array([n2, n3, n_nds+i])
        tris_conn[j+3] = np.array([n3, n0, n_nds+i])

    return tris_conn, None, None


# see also quads_to_8tris_9n
def quads_to_8tris_8n(quads_conn, nds_crd, nds_val):
    """
    Get triangles connectivity, coordinates and new values at quad centroids.

    Notes:
        Triangles connectivity array is based on
        quadrilaterals connectivity.
        Each quad is split into eight triangles.
        New nodes are created at the quad centroid.

    See also:
        function: quads_to_8tris_9n, quads_to_4tris
    """
    n_quads, _ = quads_conn.shape
    n_nds, _ = nds_crd.shape

    # coordinates and values at quad centroids _c_
    nds_c_crd = np.zeros((n_quads, 2))
    nds_c_val = np.zeros(n_quads)

    tris_conn = np.zeros((8*n_quads, 3), dtype=int)

    for i, quad_conn in enumerate(quads_conn):
        j = 8*i
        n0, n1, n2, n3, n4, n5, n6, n7 = quad_conn

        # quad centroids
        # nds_c_crd[i] = np.array([np.sum(nds_crd[[n0, n1, n2, n3], 0])/4.,
        #                          np.sum(nds_crd[[n0, n1, n2, n3], 1])/4.])
        # nds_c_val[i] = np.sum(nds_val[[n0, n1, n2, n3]])/4.
        nds_c_crd[i] = quad8n_val_at_center(nds_crd[[n0, n1, n2, n3,
                                                     n4, n5, n6, n7]])
        nds_c_val[i] = quad8n_val_at_center(nds_val[[n0, n1, n2, n3,
                                                     n4, n5, n6, n7]])

        # triangles connectivity
        tris_conn[j  ] = np.array([n0, n4, n_nds+i])
        tris_conn[j+1] = np.array([n4, n1, n_nds+i])
        tris_conn[j+2] = np.array([n1, n5, n_nds+i])
        tris_conn[j+3] = np.array([n5, n2, n_nds+i])
        tris_conn[j+4] = np.array([n2, n6, n_nds+i])
        tris_conn[j+5] = np.array([n6, n3, n_nds+i])
        tris_conn[j+6] = np.array([n3, n7, n_nds+i])
        tris_conn[j+7] = np.array([n7, n0, n_nds+i])

    return tris_conn, nds_c_crd, nds_c_val


# see also quads_to_8tris_8n
def quads_to_8tris_9n(quads_conn):
    """
    Get triangles connectivity, coordinates and new values at quad centroids.

    Args:
        quads_conn (ndarray):

    Returns:
        tris_conn, nds_c_crd, nds_c_val (tuple):

    Notes:
        Triangles connectivity array is based on
        quadrilaterals connectivity.
        Each quad is split into eight triangles.
        New nodes are created at the quad centroid.
    """
    n_quads, _ = quads_conn.shape

    tris_conn = np.zeros((8*n_quads, 3), dtype=int)

    for i, quad_conn in enumerate(quads_conn):
        j = 8*i
        n0, n1, n2, n3, n4, n5, n6, n7, n8 = quad_conn

        # triangles connectivity
        tris_conn[j]   = np.array([n0, n4, n8])
        tris_conn[j+1] = np.array([n4, n1, n8])
        tris_conn[j+2] = np.array([n1, n5, n8])
        tris_conn[j+3] = np.array([n5, n2, n8])
        tris_conn[j+4] = np.array([n2, n6, n8])
        tris_conn[j+5] = np.array([n6, n3, n8])
        tris_conn[j+6] = np.array([n3, n7, n8])
        tris_conn[j+7] = np.array([n7, n0, n8])

    return tris_conn


def bricks_to_24tris(bricks_conn, nds_crd, nds_val, disps=None):
    """
    Get triangles connectivity, coordinates and new vals at brick face centroids.

    Notes:
        Triangles connectivity array is based on
        stdBricks connectivity.
        Each of 6 brick faces is split into four triangles.
        New nodes are created at the face centroid.

    See also:
        function: bricks_to_8tris_9n, bricks_to_8tris_8n
    """
    n_bricks, _ = bricks_conn.shape
    n_nds, _ = nds_crd.shape

    # coordinates and values at brick centroids _c_
    nds_c_crd = np.zeros((n_bricks*6, 3))
    nds_c_val = np.zeros(n_bricks*6)

    disps_c = None
    if disps is not None:
        disps_c = np.zeros((n_bricks*6, 3))

    tris_conn = np.zeros((24*n_bricks, 3), dtype=int)

    for i, brick_conn in enumerate(bricks_conn):
        j = 24*i
        n0, n1, n2, n3, n4, n5, n6, n7 = brick_conn

        # brick centroids
        nds_c_crd[i*6] = np.array([np.sum(nds_crd[[n0, n1, n5, n4], 0])/4.,
                                   np.sum(nds_crd[[n0, n1, n5, n4], 1])/4.,
                                   np.sum(nds_crd[[n0, n1, n5, n5], 2])/4.])
        nds_c_crd[i*6+1] = np.array([np.sum(nds_crd[[n1, n2, n6, n5], 0])/4.,
                                     np.sum(nds_crd[[n1, n2, n6, n5], 1])/4.,
                                     np.sum(nds_crd[[n1, n2, n6, n5], 2])/4.])
        nds_c_crd[i*6+2] = np.array([np.sum(nds_crd[[n2, n3, n7, n6], 0])/4.,
                                     np.sum(nds_crd[[n2, n3, n7, n6], 1])/4.,
                                     np.sum(nds_crd[[n2, n3, n7, n6], 2])/4.])
        nds_c_crd[i*6+3] = np.array([np.sum(nds_crd[[n3, n0, n4, n7], 0])/4.,
                                     np.sum(nds_crd[[n3, n0, n4, n7], 1])/4.,
                                     np.sum(nds_crd[[n3, n0, n4, n7], 2])/4.])
        nds_c_crd[i*6+4] = np.array([np.sum(nds_crd[[n4, n5, n6, n7], 0])/4.,
                                     np.sum(nds_crd[[n4, n5, n6, n7], 1])/4.,
                                     np.sum(nds_crd[[n4, n5, n6, n7], 2])/4.])
        nds_c_crd[i*6+5] = np.array([np.sum(nds_crd[[n0, n1, n2, n3], 0])/4.,
                                     np.sum(nds_crd[[n0, n1, n2, n3], 1])/4.,
                                     np.sum(nds_crd[[n0, n1, n2, n3], 2])/4.])

        nds_c_val[6*i  ] = np.sum(nds_val[[n0, n1, n5, n4]])/4.
        nds_c_val[6*i+1] = np.sum(nds_val[[n1, n2, n6, n5]])/4.
        nds_c_val[6*i+2] = np.sum(nds_val[[n2, n3, n7, n6]])/4.
        nds_c_val[6*i+3] = np.sum(nds_val[[n3, n0, n4, n7]])/4.
        nds_c_val[6*i+4] = np.sum(nds_val[[n4, n5, n6, n7]])/4.
        nds_c_val[6*i+5] = np.sum(nds_val[[n0, n1, n2, n3]])/4.

        # triangles connectivity
        tris_conn[j  ] = np.array([n0, n1, n_nds+i*6])
        tris_conn[j+1] = np.array([n1, n5, n_nds+i*6])
        tris_conn[j+2] = np.array([n5, n4, n_nds+i*6])
        tris_conn[j+3] = np.array([n4, n0, n_nds+i*6])

        tris_conn[j+4] = np.array([n1, n2, n_nds+i*6+1])
        tris_conn[j+5] = np.array([n2, n6, n_nds+i*6+1])
        tris_conn[j+6] = np.array([n6, n5, n_nds+i*6+1])
        tris_conn[j+7] = np.array([n5, n1, n_nds+i*6+1])

        tris_conn[j+8] = np.array([n2, n3, n_nds+i*6+2])
        tris_conn[j+9] = np.array([n3, n7, n_nds+i*6+2])
        tris_conn[j+10] = np.array([n7, n6, n_nds+i*6+2])
        tris_conn[j+11] = np.array([n6, n2, n_nds+i*6+2])

        tris_conn[j+12] = np.array([n3, n0, n_nds+i*6+3])
        tris_conn[j+13] = np.array([n0, n4, n_nds+i*6+3])
        tris_conn[j+14] = np.array([n4, n7, n_nds+i*6+3])
        tris_conn[j+15] = np.array([n7, n3, n_nds+i*6+3])

        tris_conn[j+16] = np.array([n4, n5, n_nds+i*6+4])
        tris_conn[j+17] = np.array([n5, n6, n_nds+i*6+4])
        tris_conn[j+18] = np.array([n6, n7, n_nds+i*6+4])
        tris_conn[j+19] = np.array([n7, n4, n_nds+i*6+4])

        tris_conn[j+20] = np.array([n0, n1, n_nds+i*6+5])
        tris_conn[j+21] = np.array([n1, n2, n_nds+i*6+5])
        tris_conn[j+22] = np.array([n2, n3, n_nds+i*6+5])
        tris_conn[j+23] = np.array([n3, n0, n_nds+i*6+5])

        if disps is not None:
            disps_c[6*i] = np.sum(disps[[n0, n1, n5, n4]], axis=0)/4.
            disps_c[6*i+1] = np.sum(disps[[n1, n2, n6, n5]], axis=0)/4.
            disps_c[6*i+2] = np.sum(disps[[n2, n3, n7, n6]], axis=0)/4.
            disps_c[6*i+3] = np.sum(disps[[n3, n0, n4, n7]], axis=0)/4.
            disps_c[6*i+4] = np.sum(disps[[n4, n5, n6, n7]], axis=0)/4.
            disps_c[6*i+5] = np.sum(disps[[n0, n1, n2, n3]], axis=0)/4.

    return tris_conn, nds_c_crd, nds_c_val, disps_c


# brick20n bricks to tris
def bricks_to_48tris(bricks_conn, nds_crd, nds_val, disps=None):
    """
    Get triangles connectivity, coordinates and new vals at brick face centroids,
    for brick20n

    Notes:
        Triangles connectivity array is based on
        stdBricks connectivity.
        Each of 6 brick faces is split into four triangles.
        New nodes are created at the face centroid.

    See also:
        function: bricks_to_24tris
    """
    n_bricks, _ = bricks_conn.shape
    n_nds, _ = nds_crd.shape

    # coordinates and values at brick centroids _c_
    nds_c_crd = np.zeros((n_bricks*6, 3))
    nds_c_val = np.zeros(n_bricks*6)

    disps_c = None
    if disps is not None:
        disps_c = np.zeros((n_bricks*6, 3))

    tris_conn = np.zeros((48*n_bricks, 3), dtype=int)

    for i, brick_conn in enumerate(bricks_conn):
        j = 48*i
        n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, \
            n14, n15, n16, n17, n18, n19 = brick_conn

        # brick centroids
        nds_c_crd[i*6] = np.array([np.sum(nds_crd[[n0, n1, n5, n4], 0])/4.,
                                   np.sum(nds_crd[[n0, n1, n5, n4], 1])/4.,
                                   np.sum(nds_crd[[n0, n1, n5, n5], 2])/4.])
        nds_c_crd[i*6+1] = np.array([np.sum(nds_crd[[n1, n2, n6, n5], 0])/4.,
                                     np.sum(nds_crd[[n1, n2, n6, n5], 1])/4.,
                                     np.sum(nds_crd[[n1, n2, n6, n5], 2])/4.])
        nds_c_crd[i*6+2] = np.array([np.sum(nds_crd[[n2, n3, n7, n6], 0])/4.,
                                     np.sum(nds_crd[[n2, n3, n7, n6], 1])/4.,
                                     np.sum(nds_crd[[n2, n3, n7, n6], 2])/4.])
        nds_c_crd[i*6+3] = np.array([np.sum(nds_crd[[n3, n0, n4, n7], 0])/4.,
                                     np.sum(nds_crd[[n3, n0, n4, n7], 1])/4.,
                                     np.sum(nds_crd[[n3, n0, n4, n7], 2])/4.])
        nds_c_crd[i*6+4] = np.array([np.sum(nds_crd[[n4, n5, n6, n7], 0])/4.,
                                     np.sum(nds_crd[[n4, n5, n6, n7], 1])/4.,
                                     np.sum(nds_crd[[n4, n5, n6, n7], 2])/4.])
        nds_c_crd[i*6+5] = np.array([np.sum(nds_crd[[n0, n1, n2, n3], 0])/4.,
                                     np.sum(nds_crd[[n0, n1, n2, n3], 1])/4.,
                                     np.sum(nds_crd[[n0, n1, n2, n3], 2])/4.])

        nds_c_val[6*i  ] = np.sum(nds_val[[n0, n1, n5, n4]])/4.
        nds_c_val[6*i+1] = np.sum(nds_val[[n1, n2, n6, n5]])/4.
        nds_c_val[6*i+2] = np.sum(nds_val[[n2, n3, n7, n6]])/4.
        nds_c_val[6*i+3] = np.sum(nds_val[[n3, n0, n4, n7]])/4.
        nds_c_val[6*i+4] = np.sum(nds_val[[n4, n5, n6, n7]])/4.
        nds_c_val[6*i+5] = np.sum(nds_val[[n0, n1, n2, n3]])/4.

        # triangles connectivity
        tris_conn[j  ] = np.array([n0, n8, n_nds+i*6])
        tris_conn[j+1] = np.array([n8, n1, n_nds+i*6])
        tris_conn[j+2] = np.array([n1, n17, n_nds+i*6])
        tris_conn[j+3] = np.array([n17, n5, n_nds+i*6])
        tris_conn[j+4] = np.array([n5, n12, n_nds+i*6])
        tris_conn[j+5] = np.array([n12, n4, n_nds+i*6])
        tris_conn[j+6] = np.array([n4, n16, n_nds+i*6])
        tris_conn[j+7] = np.array([n16, n0, n_nds+i*6])

        tris_conn[j+8] = np.array([n1, n9, n_nds+i*6+1])
        tris_conn[j+9] = np.array([n9, n2, n_nds+i*6+1])
        tris_conn[j+10] = np.array([n2, n18, n_nds+i*6+1])
        tris_conn[j+11] = np.array([n18, n6, n_nds+i*6+1])
        tris_conn[j+12] = np.array([n6, n13, n_nds+i*6+1])
        tris_conn[j+13] = np.array([n13, n5, n_nds+i*6+1])
        tris_conn[j+14] = np.array([n5, n17, n_nds+i*6+1])
        tris_conn[j+15] = np.array([n17, n1, n_nds+i*6+1])

        tris_conn[j+16] = np.array([n2, n10, n_nds+i*6+2])
        tris_conn[j+17] = np.array([n10, n3, n_nds+i*6+2])
        tris_conn[j+18] = np.array([n3, n19, n_nds+i*6+2])
        tris_conn[j+19] = np.array([n19, n7, n_nds+i*6+2])
        tris_conn[j+20] = np.array([n7, n14, n_nds+i*6+2])
        tris_conn[j+21] = np.array([n14, n6, n_nds+i*6+2])
        tris_conn[j+22] = np.array([n6, n18, n_nds+i*6+2])
        tris_conn[j+23] = np.array([n18, n2, n_nds+i*6+2])

        tris_conn[j+24] = np.array([n3, n11, n_nds+i*6+3])
        tris_conn[j+25] = np.array([n11, n0, n_nds+i*6+3])
        tris_conn[j+26] = np.array([n0, n16, n_nds+i*6+3])
        tris_conn[j+27] = np.array([n16, n4, n_nds+i*6+3])
        tris_conn[j+28] = np.array([n4, n15, n_nds+i*6+3])
        tris_conn[j+29] = np.array([n15, n7, n_nds+i*6+3])
        tris_conn[j+30] = np.array([n7, n19, n_nds+i*6+3])
        tris_conn[j+31] = np.array([n19, n3, n_nds+i*6+3])

        tris_conn[j+32] = np.array([n4, n12, n_nds+i*6+4])
        tris_conn[j+33] = np.array([n12, n5, n_nds+i*6+4])
        tris_conn[j+34] = np.array([n5, n13, n_nds+i*6+4])
        tris_conn[j+35] = np.array([n13, n6, n_nds+i*6+4])
        tris_conn[j+36] = np.array([n6, n14, n_nds+i*6+4])
        tris_conn[j+37] = np.array([n14, n7, n_nds+i*6+4])
        tris_conn[j+38] = np.array([n7, n15, n_nds+i*6+4])
        tris_conn[j+39] = np.array([n15, n4, n_nds+i*6+4])

        tris_conn[j+40] = np.array([n0, n8, n_nds+i*6+5])
        tris_conn[j+41] = np.array([n8, n1, n_nds+i*6+5])
        tris_conn[j+42] = np.array([n1, n9, n_nds+i*6+5])
        tris_conn[j+43] = np.array([n9, n2, n_nds+i*6+5])
        tris_conn[j+44] = np.array([n2, n10, n_nds+i*6+5])
        tris_conn[j+45] = np.array([n10, n3, n_nds+i*6+5])
        tris_conn[j+46] = np.array([n3, n11, n_nds+i*6+5])
        tris_conn[j+47] = np.array([n11, n0, n_nds+i*6+5])

        if disps is not None:
            disps_c[6*i] = np.sum(disps[[n0, n1, n5, n4,
                                         n8, n17, n12, n16]], axis=0)/4.
            disps_c[6*i+1] = np.sum(disps[[n1, n2, n6, n5,
                                           n9, n18, n13, n17]], axis=0)/4.
            disps_c[6*i+2] = np.sum(disps[[n2, n3, n7, n6,
                                           n10, n19, n14, n18]], axis=0)/4.
            disps_c[6*i+3] = np.sum(disps[[n3, n0, n4, n7,
                                           n11, n16, n15, n19]], axis=0)/4.
            disps_c[6*i+4] = np.sum(disps[[n4, n5, n6, n7,
                                           n12, n13, n14, n15]], axis=0)/4.
            disps_c[6*i+5] = np.sum(disps[[n0, n1, n2, n3,
                                           n8, n9, n10, n11]], axis=0)/4.

    return tris_conn, nds_c_crd, nds_c_val, disps_c


def tetra4n_to_4tris(tetras4n_conn):
    """Get triangles connectivity.

    Four-node tetrahedron is subdivided into four triangles

    Args:
        tetra4n_conn (ndarray):

    Returns:
        tris_conn_subdiv (ndarray):
    """
    n_tetras, _ = tetras4n_conn.shape
    # n_nds, _ = nds_crd.shape

    tris_conn_subdiv = np.zeros((4*n_tetras, 3), dtype=int)

    for i, tetra4n_conn in enumerate(tetras4n_conn):
        j = 4*i
        n0, n1, n2, n3 = tetra4n_conn

        # triangles connectivity
        tris_conn_subdiv[j] = np.array([n0, n1, n2])
        tris_conn_subdiv[j+1] = np.array([n0, n3, n1])
        tris_conn_subdiv[j+2] = np.array([n0, n2, n3])
        tris_conn_subdiv[j+3] = np.array([n1, n2, n3])

    return tris_conn_subdiv


def tetra10n_to_16tris(tetras10n_conn):
    """Get triangles connectivity.

    Six-node triangle is subdivided into four triangles

    Args:
        tetra10n_conn (ndarray):

    Returns:
        tris_conn_subdiv (ndarray):
    """
    n_tetras, _ = tetras10n_conn.shape
    # n_nds, _ = nds_crd.shape

    tris_conn_subdiv = np.zeros((16*n_tetras, 3), dtype=int)

    for i, tetra10n_conn in enumerate(tetras10n_conn):
        j = 16*i
        n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = tetra10n_conn

        # triangles connectivity
        tris_conn_subdiv[j] = np.array([n0, n4, n6])
        tris_conn_subdiv[j+1] = np.array([n1, n5, n4])
        tris_conn_subdiv[j+2] = np.array([n4, n5, n6])
        tris_conn_subdiv[j+3] = np.array([n2, n6, n5])

        tris_conn_subdiv[j+4] = np.array([n0, n7, n4])
        tris_conn_subdiv[j+5] = np.array([n4, n7, n8])
        tris_conn_subdiv[j+6] = np.array([n1, n4, n8])
        tris_conn_subdiv[j+7] = np.array([n3, n8, n7])

        tris_conn_subdiv[j+8] = np.array([n0, n6, n7])
        tris_conn_subdiv[j+9] = np.array([n6, n9, n7])
        tris_conn_subdiv[j+10] = np.array([n2, n9, n6])
        tris_conn_subdiv[j+11] = np.array([n3, n7, n9])

        tris_conn_subdiv[j+12] = np.array([n1, n5, n8])
        tris_conn_subdiv[j+13] = np.array([n5, n9, n8])
        tris_conn_subdiv[j+14] = np.array([n2, n9, n5])
        tris_conn_subdiv[j+15] = np.array([n3, n8, n9])

    return tris_conn_subdiv


def tris6n_to_4tris(tris_conn):
    """Get triangles connectivity.

    Six-node triangle is subdivided into four triangles
    """
    n_tris, _ = tris_conn.shape
    # n_nds, _ = nds_crd.shape

    tris_conn_subdiv = np.zeros((4*n_tris, 3), dtype=int)

    for i, tri_conn in enumerate(tris_conn):
        j = 4*i
        n0, n1, n2, n3, n4, n5 = tri_conn

        # triangles connectivity
        tris_conn_subdiv[j  ] = np.array([n0, n3, n5])
        tris_conn_subdiv[j+1] = np.array([n3, n1, n4])
        tris_conn_subdiv[j+2] = np.array([n3, n4, n5])
        tris_conn_subdiv[j+3] = np.array([n5, n4, n2])

    return tris_conn_subdiv
