
def draw_extrusions(model, canvas, state=None, config=None, Ra=None):
    ndm = 3

    coords = [] # Global mesh coordinates
    triang = []
    caps   = []
    locoor = [] # Local mesh coordinates, used for textures

    if config is None:
        config = {
                "style": MeshStyle(color="gray")
        }

    scale_section = config["scale"]


    I = 0
    # Track outlines with excessive edges (eg, circles) to later avoid showing
    # their edges
    no_outline = set()
    for tag in model.iter_cell_tags():

        section = model.frame_section(tag)
        if section is None:
            continue

        outline_scale = scale_section

        nen  = len(model.cell_nodes(tag))

        Xi = model.cell_position(tag)
        if state is not None:
            glob_displ = state.cell_array(tag, state.position)
            X = shps.curve.displace(Xi, glob_displ, nen).T
            R = state.cell_array(tag, state.rotation)
        else:
            outline_scale *= 0.99
            X = np.array(Xi)
            R = [model.frame_orientation(tag).T]*nen


        noe = len(section.exterior())
        try:
            face_i = model.frame_section(tag, 0).exterior()[:,1:]
            face_j = model.frame_section(tag, 1).exterior()[:,1:]
            caps.append(I+np.array(earcut(face_i)))
            caps.append(I+(nen-1)*noe + np.array(earcut(face_j)))
        except Exception as e:
            warnings.warn(f"Earcut failed with message: {e}")

        # Loop over sample points along element length to assemble
        # `coord` and `triang` arrays
        for j in range(nen):
            section = model.frame_section(tag, j) # TODO: Pass float between 0 and 1 instead of j
            outline = section.exterior().copy()
            outline[:,1:] *= outline_scale
            # Loop over section edges
            for k,edge in enumerate(outline):
                # Append rotated section coordinates to list of coordinates
                coords.append(X[j, :] + R[j]@edge)
                locoor.append([ (j+0)/nen+0.1,  0.1+(k+0)/(noe+0) ])

                if j == 0:
                    # Skip the first section
                    continue

                elif k < noe-1:
                    triang.extend([
                        [I+    noe*j + k,   I+    noe*j + k + 1,    I+noe*(j-1) + k],
                        [I+noe*j + k + 1,   I+noe*(j-1) + k + 1,    I+noe*(j-1) + k]
                    ])
                else:
                    # elif j < N-1:
                    triang.extend([
                        [I+    noe*j + k,    I + noe*j , I+noe*(j-1) + k],
                        [      I + noe*j, I + noe*(j-1), I+noe*(j-1) + k]
                    ])

                if len(outline) > 25:
                    no_outline.add(len(triang)-1)
                    no_outline.add(len(triang)-2)

        I += nen*noe

    triang = [list(reversed(i)) for i in triang]

    if len(triang) == 0:
        return

    mesh = canvas.plot_mesh(coords, triang, local_coords=locoor, style=config["style"])

    if len(caps) > 0:
        for cap in caps:
            try:
                canvas.plot_mesh(mesh.vertices, cap, style=config["style"])
            except:
                pass

    IDX = np.array((
        (0, 2),
        (0, 1)
    ))

    triang = [list(reversed(i)) for i in triang]

    nan = np.zeros(ndm)*np.nan
    coords = np.array(coords)
    if "tran" in config["outline"]:
        tri_points = np.array([
            coords[idx]  if (j+1)%3 else nan
            for j,idx in enumerate(np.array(triang).reshape(-1))
        ])
    elif "long" in config["outline"]:
        tri_points = np.array([
            coords[i]  if j%2 else nan
            for j,idx in enumerate(np.array(triang)) for i in idx[IDX[j%2]] if j not in no_outline
        ])
    else:
        return

    canvas.plot_lines(tri_points,
                      style=config["line_style"]
    )

def draw_extrusions2(model, canvas, state=None, config=None, Ra=None):
    ndm = 3

    coords = [] # Global mesh coordinates
    triang = []
    caps   = []
    locoor = [] # Local mesh coordinates, used for textures

    if config is None:
        config = {
                "style": MeshStyle(color="gray")
        }

    scale_section = config["scale"]


    I = 0
    # Track outlines with excessive edges (eg, circles) to later avoid showing
    # their edges
    no_outline = set()
    for tag in model.iter_cell_tags():

        section = model.frame_section(tag)
        if section is None:
            continue

        outline_scale = scale_section

        nen  = len(model.cell_nodes(tag))

        Xi = model.cell_position(tag)
        if state is not None:
            glob_displ = state.cell_array(tag, state.position)
            X = shps.curve.displace(Xi, glob_displ, nen).T
            R = state.cell_array(tag, state.rotation)
        else:
            outline_scale *= 0.99
            X = np.array(Xi)
            R = [model.frame_orientation(tag).T]*nen


        noe = len(section.exterior())
        try:
            face_i = model.frame_section(tag, 0).exterior()[:,1:]
            face_j = model.frame_section(tag, 1).exterior()[:,1:]
            caps.append(I+np.array(earcut(face_i)))
            caps.append(I+(nen-1)*noe + np.array(earcut(face_j)))
        except Exception as e:
            warnings.warn(f"Earcut failed with message: {e}")

        # Loop over sample points along element length to assemble
        # `coord` and `triang` arrays
        for j in range(nen):
            section = model.frame_section(tag, j) # TODO: Pass float between 0 and 1 instead of j
            outline = section.exterior().copy()
            outline[:,1:] *= outline_scale
            # Loop over section edges
            for k,edge in enumerate(outline):
                # Append rotated section coordinates to list of coordinates
                coords.append(X[j, :] + R[j]@edge)
                locoor.append([ (j+0)/nen+0.1,  0.1+(k+0)/(noe+0) ])

                if j == 0:
                    # Skip the first section
                    continue

                elif k < noe-1:
                    triang.extend([
                        [I+    noe*j + k,   I+    noe*j + k + 1,    I+noe*(j-1) + k],
                        [I+noe*j + k + 1,   I+noe*(j-1) + k + 1,    I+noe*(j-1) + k]
                    ])
                else:
                    # elif j < N-1:
                    triang.extend([
                        [I+    noe*j + k,    I + noe*j , I+noe*(j-1) + k],
                        [      I + noe*j, I + noe*(j-1), I+noe*(j-1) + k]
                    ])

                if len(outline) > 25:
                    no_outline.add(len(triang)-1)
                    no_outline.add(len(triang)-2)

        I += nen*noe

    triang = [list(reversed(i)) for i in triang]

    if len(triang) == 0:
        return

    mesh = canvas.plot_mesh(coords, triang, local_coords=locoor, style=config["style"])

    if len(caps) > 0:
        for cap in caps:
            try:
                canvas.plot_mesh(mesh.vertices, cap, style=config["style"])
            except:
                pass


    IDX = np.array((
        (0, 2),
        (0, 1)
    ))

    triang = [list(reversed(i)) for i in triang]

    nan = np.zeros(ndm)*np.nan
    coords = np.array(coords)
    if "tran" in config["outline"]:
        tri_points = np.array([
            coords[idx]  if (j+1)%3 else nan
            for j,idx in enumerate(np.array(triang).reshape(-1))
        ])
    elif "long" in config["outline"]:
        tri_points = np.array([
            coords[i]  if j%2 else nan
            for j,idx in enumerate(np.array(triang)) for i in idx[IDX[j%2]] if j not in no_outline
        ])
    else:
        return

    if len(tri_points):
        canvas.plot_lines(tri_points,
                        style=config["line_style"]
        )
