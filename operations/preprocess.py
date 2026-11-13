def find_degenerate_corners(patch_input_data):
    degenerate_edge_id_set = set()
    patch_degenerate_corner_id_set = set()
    for loop_id, loop_geometry_data in patch_input_data.patch_loop_geometry_data.items():
        for feature_id, poly_V in loop_geometry_data.feature_id_poly_V_True_orientation.items():
            if len(poly_V) == 0:
                # this is a degenerate corner
                degenerate_edge_id_set.add(feature_id)

    for loop_id, loop_topology_data in patch_input_data.patch_loop_topology_data.items():
        for i, halfedge in enumerate(loop_topology_data.halfedges):
            if halfedge.feature_id in degenerate_edge_id_set:
                halfedge.is_degenerate = True
                patch_degenerate_corner_id_set.add(halfedge.start_corner_id)
                patch_degenerate_corner_id_set.add(halfedge.end_corner_id)

    return patch_degenerate_corner_id_set
