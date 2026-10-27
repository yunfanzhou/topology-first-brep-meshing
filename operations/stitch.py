
from utils.mesh_utils import create_edge
import numpy as np
from utils.print import print_if_verbose
from utils.dijkstra import bfs
from operations.internal_operations import split_edge_in_mesh_overloaded


def pre_split_halfcorners(V, connectivity, halfcorner_vids, prohibited_vids=None):
    """Pre-split between every pair of halfcorner vids before stitching, so no
    face spans two of them (which would collapse to zero area on merge).

    Caller is responsible for syncing any auxiliary V structures (e.g. V_uv)
    with the new midpoints appended to V.
    """
    if len(halfcorner_vids) < 2:
        return
    prohibited = set(halfcorner_vids)
    if prohibited_vids:
        prohibited |= set(prohibited_vids)
    for i in range(len(halfcorner_vids)):
        for j in range(i + 1, len(halfcorner_vids)):
            split_between_two_stitching_vertices(V, connectivity,
                                                 halfcorner_vids[i],
                                                 halfcorner_vids[j],
                                                 prohibited)


def stitch_halfedges_same_patch(V,
                                connectivity,
                                trace0,
                                trace1):
    '''
    Trace0 and trace1 are already aligned. trace0 and trace1 are also in same orientation
    Stitching the two feature edges to the midpoint.
    Return the stitched trace that is in True orientation.
    Meanwhile, we need to update the connectivity data
    Note: we do not stitch the two end points. Because corners might have high valence of halfedges.
    For example, a torus can have a corner with 4 halfedges. To keep the corner id to edge id consistent, we do a stitching of all the hafedges.

    Note: pre-splitting to prevent zero-area faces on merge is the caller's
    responsibility (it grows V, so the caller must keep any auxiliary V
    structures — e.g. V_uv — in sync).
    '''
    # stitch the two feature edges
    for i in range(1, len(trace0)-1):
        # ensure arithmetic works whether V stores lists or numpy arrays
        mid_point = (np.asarray(V[trace0[i]]) + np.asarray(V[trace1[i]])) / 2.0
        # store back as native Python list to keep V mutable in caller code
        V[trace0[i]] = mid_point.tolist()
        # Store the old adjacencies for trace1[i] before removing them
        old_adjacencies_trace1 = set(connectivity.adj_list[trace1[i]])
        # Update faces that reference trace1[i] to reference trace0[i]
        for f in connectivity.VF[trace1[i]]:
            for j in range(3):
                if connectivity.F[f][j] == trace1[i]:
                    connectivity.F[f][j] = trace0[i]
                    # update VF for trace0[i]
                    connectivity.VF[trace0[i]].append(f)
                    break
        
        # Remove VF entry for trace1[i] since it's being merged
        connectivity.VF.pop(trace1[i], None)
        
        # Update adjacency list: merge trace1[i]'s adjacencies into trace0[i]
        for neighbor in old_adjacencies_trace1:
            if neighbor != trace0[i]:  # avoid self-loop
                # Add to trace0[i]'s adjacency list
                connectivity.adj_list[trace0[i]].add(neighbor)
                # Update neighbor's adjacency list to point to trace0[i] instead of trace1[i]
                connectivity.adj_list[neighbor].remove(trace1[i])
                connectivity.adj_list[neighbor].add(trace0[i])
        
        # Remove edge length mappings for trace1[i] using the old adjacencies
        for neighbor in old_adjacencies_trace1:
            edge = create_edge(trace1[i], neighbor)
            connectivity.edge_set_to_length_mapping.pop(edge, None)
        
        # Clear trace1[i]'s adjacency list since it's being removed
        connectivity.adj_list[trace1[i]].clear()
        # Update edge lengths only for modified edges
    for i in range(1, len(trace0)-1):
        # Update edge lengths for trace0[i] with all its neighbors
        for n in connectivity.adj_list[trace0[i]]:
            edge = create_edge(trace0[i], n)
            edge_length = np.linalg.norm(np.asarray(V[trace0[i]]) - np.asarray(V[n]))
            connectivity.edge_set_to_length_mapping[edge] = edge_length
    return trace0

def stitch_halfcorners_same_patch(V,
                                  connectivity,
                                  halfcorner_vids : list[int]):
    '''
    Stitch multiple corner vertices (halfcorners) to their centroid position.
    All vertices in halfcorner_vids represent the same geometric corner but are
    separate due to topology. We merge them all into the first vertex.

    Note: pre-splitting to prevent zero-area faces on merge is the caller's
    responsibility (it grows V, so the caller must keep any auxiliary V
    structures — e.g. V_uv — in sync). See pre_split_halfcorners above.
    '''
    if len(halfcorner_vids) < 2:
        return halfcorner_vids[0] if halfcorner_vids else None

    V_np = np.array(V)
    avg_position = np.mean(V_np[halfcorner_vids], axis=0)
    stitched_to_vid = halfcorner_vids[0]
    
    # Update the position of the target vertex to the centroid
    V[stitched_to_vid] = avg_position.tolist()
    
    # Merge all other corner vertices into the first one
    for i in range(1, len(halfcorner_vids)):
        vid_to_merge = halfcorner_vids[i]
        # Store old adjacencies before modifying
        old_adjacencies = set(connectivity.adj_list[vid_to_merge])
        # Update faces that reference vid_to_merge to reference stitched_to_vid
        if vid_to_merge in connectivity.VF:
            for f in connectivity.VF[vid_to_merge]:
                for j in range(3):
                    if connectivity.F[f][j] == vid_to_merge:
                        connectivity.F[f][j] = stitched_to_vid
                        # Add face to stitched_to_vid's VF list
                        if stitched_to_vid not in connectivity.VF:
                            raise Exception("This should not happen, stitched_to_vid in stitching_halfcorners must be in VF")
                        connectivity.VF[stitched_to_vid].append(f)
                        break
            
            # Remove VF entry for the merged vertex
            connectivity.VF.pop(vid_to_merge, None)

        # Update adjacency lists: merge vid_to_merge's adjacencies into stitched_to_vid
        for neighbor in old_adjacencies:
            if neighbor != stitched_to_vid and neighbor not in halfcorner_vids:
                # Add to stitched_to_vid's adjacency list
                connectivity.adj_list[stitched_to_vid].add(neighbor)
                # Update neighbor's adjacency list
                if vid_to_merge in connectivity.adj_list[neighbor]:
                    connectivity.adj_list[neighbor].remove(vid_to_merge)
                connectivity.adj_list[neighbor].add(stitched_to_vid)
        
        # Remove edge length mappings for vid_to_merge
        for neighbor in old_adjacencies:
            edge = create_edge(vid_to_merge, neighbor)
            connectivity.edge_set_to_length_mapping.pop(edge, None)
        
        # Clear the merged vertex's adjacency list
        connectivity.adj_list[vid_to_merge].clear()
    # Update edge lengths for the stitched vertex with all its neighbors
    for neighbor in connectivity.adj_list[stitched_to_vid]:
        edge = create_edge(stitched_to_vid, neighbor)
        edge_length = np.linalg.norm(np.asarray(V[stitched_to_vid]) - np.asarray(V[neighbor]))
        connectivity.edge_set_to_length_mapping[edge] = edge_length
    return stitched_to_vid

def stitch_periodic_edges(V_3d_list,
                          connectivity,
                          u_periodic_edges,
                          v_periodic_edges):
    """Stitch periodic seam edges/corners and remove degenerate/non-manifold faces.

    Parameters
    ----------
    V_3d_list : list of 3D vertex positions (modified in place by stitching).
    connectivity : ConnectivityMaintainance (F modified in place).
    u_periodic_edges, v_periodic_edges : list of (side0, side1) chain pairs.

    Returns
    -------
    merged_u_vertices : list of merged u-seam vertex ids.
    merged_v_vertices : list of merged v-seam vertex ids.
    stitched_to_crux_vid : int or None (double-periodic corner).
    """
    merged_u_vertices = []
    merged_v_vertices = []
    has_u = len(u_periodic_edges) > 0
    has_v = len(v_periodic_edges) > 0

    # Stitch each chain pair independently
    if has_u:
        for (edges0, edges1) in u_periodic_edges:
            merged = stitch_halfedges_same_patch(
                V_3d_list, connectivity, edges0, edges1)
            merged_u_vertices.extend(merged)
    if has_v:
        for (edges0, edges1) in v_periodic_edges:
            merged = stitch_halfedges_same_patch(
                V_3d_list, connectivity, edges0, edges1)
            merged_v_vertices.extend(merged)

    stitched_to_crux_vid = None

    # Corner stitching: each chain pair stitches its own endpoints
    if has_u:
        for (edges0, edges1) in u_periodic_edges:
            if len(edges0) > 0 and len(edges1) > 0:
                stitch_halfcorners_same_patch(
                    V_3d_list, connectivity,
                    [edges0[0], edges1[0]])
                stitch_halfcorners_same_patch(
                    V_3d_list, connectivity,
                    [edges0[-1], edges1[-1]])
    if has_v:
        for (edges0, edges1) in v_periodic_edges:
            if len(edges0) > 0 and len(edges1) > 0:
                stitch_halfcorners_same_patch(
                    V_3d_list, connectivity,
                    [edges0[0], edges1[0]])
                stitch_halfcorners_same_patch(
                    V_3d_list, connectivity,
                    [edges0[-1], edges1[-1]])

    # Double-periodic corner: stitch all 4 global corners if both u and v periodic
    if has_u and has_v:
        all_corner_vids = []
        all_corner_vids.append(u_periodic_edges[0][0][0])
        all_corner_vids.append(u_periodic_edges[0][1][0])
        all_corner_vids.append(u_periodic_edges[-1][0][-1])
        all_corner_vids.append(u_periodic_edges[-1][1][-1])
        stitched_to_crux_vid = stitch_halfcorners_same_patch(
            V_3d_list, connectivity, all_corner_vids)

    return (merged_u_vertices,
            merged_v_vertices,
            stitched_to_crux_vid)

def stitch_periodic_edges_with_split(V_3d, connectivity_3d, u_periodic_edges, v_periodic_edges, singularity_edges=None):
    """Stitch periodic seam edges with pre-split for manifoldness.

    Parameters
    ----------
    u_periodic_edges, v_periodic_edges : list of (side0, side1) chain pairs.
    """
    # 1. stitching periodic edges
    #   1.1 First, we split to ensure after stitching the mesh is valid and is manifold.
    #   1.2 Then we can stitch the periodic edges (without stitching the corners)

    # Collect all prohibited vids from all chain pairs
    prohibited_vids_set = set()
    for (edges0, edges1) in u_periodic_edges:
        prohibited_vids_set.update(edges0)
        prohibited_vids_set.update(edges1)
    for (edges0, edges1) in v_periodic_edges:
        prohibited_vids_set.update(edges0)
        prohibited_vids_set.update(edges1)
    merged_u_vertices = list()
    merged_u_sub_chains = []  # list of per-sub-chain merged vertex lists
    has_u = len(u_periodic_edges) > 0
    has_v = len(v_periodic_edges) > 0

    if has_u:
        for (edges0, edges1) in u_periodic_edges:
            assert len(edges0) == len(edges1), "u periodic edges length mismatch"
            for i in range(len(edges0)):
                vid0 = edges0[i]
                vid1 = edges1[i]
                split_between_two_stitching_vertices_updating_sing(V_3d,
                                                     connectivity_3d,
                                                     vid0, vid1,
                                                     prohibited_vids_set,
                                                     singularity_edges)
            merged = stitch_halfedges_same_patch(V_3d, connectivity_3d, edges0, edges1)
            merged_u_sub_chains.append(merged)
            merged_u_vertices.extend(merged)

    merged_v_vertices = list()
    merged_v_sub_chains = []  # list of per-sub-chain merged vertex lists
    if has_v:
        for (edges0, edges1) in v_periodic_edges:
            assert len(edges0) == len(edges1), "v periodic edges length mismatch"
            for i in range(len(edges0)):
                vid0 = edges0[i]
                vid1 = edges1[i]
                split_between_two_stitching_vertices_updating_sing(V_3d,
                                                     connectivity_3d,
                                                     vid0, vid1,
                                                     prohibited_vids_set,
                                                     singularity_edges)
            merged = stitch_halfedges_same_patch(V_3d, connectivity_3d, edges0, edges1)
            merged_v_sub_chains.append(merged)
            merged_v_vertices.extend(merged)

    #   1.3 Corner stitching: each chain pair is an independent sub-mesh
    stitched_to_crux_vid = None

    # Per-pair corner stitching
    for (edges0, edges1) in u_periodic_edges:
        if len(edges0) > 0 and len(edges1) > 0:
            stitch_halfcorners_same_patch(V_3d, connectivity_3d,
                                         [edges0[0], edges1[0]])
            stitch_halfcorners_same_patch(V_3d, connectivity_3d,
                                         [edges0[-1], edges1[-1]])
    for (edges0, edges1) in v_periodic_edges:
        if len(edges0) > 0 and len(edges1) > 0:
            stitch_halfcorners_same_patch(V_3d, connectivity_3d,
                                         [edges0[0], edges1[0]])
            stitch_halfcorners_same_patch(V_3d, connectivity_3d,
                                         [edges0[-1], edges1[-1]])

    # Double-periodic: stitch global corners
    if has_u and has_v:
        all_corner_vids = []
        all_corner_vids.append(u_periodic_edges[0][0][0])
        all_corner_vids.append(u_periodic_edges[0][1][0])
        all_corner_vids.append(u_periodic_edges[-1][0][-1])
        all_corner_vids.append(u_periodic_edges[-1][1][-1])
        stitched_to_crux_vid = stitch_halfcorners_same_patch(
            V_3d, connectivity_3d, all_corner_vids)

    # Remap stale vertex ids after stitching: every vid merged away by the seam or
    # corner stitches maps to its survivor (transitively: a corner is merged by its
    # pair stitch and again into the crux). Applied to singularity_edges and to the
    # merged seam chains, which become surface_merged_edges (the frozen seam barrier):
    # a stale id there leaves the real seam edge at that corner unfrozen.
    remap = {}
    for (edges0, edges1) in u_periodic_edges:
        for i in range(1, len(edges1) - 1):
            remap[edges1[i]] = edges0[i]
    for (edges0, edges1) in v_periodic_edges:
        for i in range(1, len(edges1) - 1):
            remap[edges1[i]] = edges0[i]
    # Corner merges
    for (edges0, edges1) in u_periodic_edges:
        if len(edges0) > 0 and len(edges1) > 0:
            if edges1[0] != edges0[0]:
                remap[edges1[0]] = edges0[0]
            if edges1[-1] != edges0[-1]:
                remap[edges1[-1]] = edges0[-1]
    for (edges0, edges1) in v_periodic_edges:
        if len(edges0) > 0 and len(edges1) > 0:
            if edges1[0] != edges0[0]:
                remap[edges1[0]] = edges0[0]
            if edges1[-1] != edges0[-1]:
                remap[edges1[-1]] = edges0[-1]
    if stitched_to_crux_vid is not None:
        for (edges0, edges1) in u_periodic_edges:
            for vid in [edges0[0], edges1[0], edges0[-1], edges1[-1]]:
                if vid != stitched_to_crux_vid:
                    remap[vid] = stitched_to_crux_vid
    def resolve(v):
        seen = set()
        while v in remap and v not in seen and remap[v] != v:
            seen.add(v)
            v = remap[v]
        return v
    if singularity_edges is not None:
        for idx, edges in singularity_edges.items():
            for i, e in enumerate(edges):
                a, b = resolve(e[0]), resolve(e[1])
                if a != e[0] or b != e[1]:
                    edges[i] = (a, b)
    merged_u_sub_chains = [[resolve(v) for v in chain] for chain in merged_u_sub_chains]
    merged_v_sub_chains = [[resolve(v) for v in chain] for chain in merged_v_sub_chains]
    merged_u_vertices = [resolve(v) for v in merged_u_vertices]
    merged_v_vertices = [resolve(v) for v in merged_v_vertices]

    # Clean up stale connectivity temp data
    connectivity_3d.adj_list_copy_dict = {}
    connectivity_3d.modified_edge_set = set()
    return merged_u_vertices, merged_v_vertices, stitched_to_crux_vid, merged_u_sub_chains, merged_v_sub_chains


def build_surface_merged_edges_with_apex(merged_u_sub_chains, merged_v_sub_chains, face, sing_info):
    """Build surface_merged_edges by inserting singularity apex vertices into merged chains.

    For boundary singularity: apex becomes one endpoint of the chain.
    For interior singularity: apex bridges consecutive sub-chains.
    Handles mixed cases (both boundary and interior on same direction).

    Parameters
    ----------
    merged_u_sub_chains : list of per-sub-chain merged vertex lists for u-periodic.
    merged_v_sub_chains : list of per-sub-chain merged vertex lists for v-periodic.
    face : abs Face object with singularities.
    sing_info : dict mapping singularity idx -> {"center_vertex_id": int}.

    Returns
    -------
    surface_merged_edges : list of (vid_a, vid_b) edge tuples.
    """
    surface_merged_edges = []

    if not face.singularities or not sing_info:
        for sub_chains in (merged_u_sub_chains, merged_v_sub_chains):
            for chain in sub_chains:
                for i in range(len(chain) - 1):
                    surface_merged_edges.append(create_edge(chain[i], chain[i + 1]))
        return surface_merged_edges

    # Classify apexes by direction and boundary/interior:
    # u-iso singularity affects v_periodic sub_chains (seam runs along u)
    # v-iso singularity affects u_periodic sub_chains (seam runs along v)
    u_iso_lower = []   # boundary apexes at lower u (affect v_periodic)
    u_iso_upper = []   # boundary apexes at upper u (affect v_periodic)
    u_iso_interior = []  # interior apexes sorted by u position (affect v_periodic)
    v_iso_lower = []   # boundary apexes at lower v (affect u_periodic)
    v_iso_upper = []   # boundary apexes at upper v (affect u_periodic)
    v_iso_interior = []  # interior apexes sorted by v position (affect u_periodic)

    for idx, info in sing_info.items():
        apex_vid = info.get("center_vertex_id", -1)
        if apex_vid < 0:
            continue
        sing = face.singularities[idx]
        boundary_side = info.get('boundary_side')
        pos = np.asarray(sing['first2d'], dtype=float)

        if sing['uiso']:
            if boundary_side == 'left':
                u_iso_lower.append(apex_vid)
            elif boundary_side == 'right':
                u_iso_upper.append(apex_vid)
            else:
                u_iso_interior.append((pos[0], apex_vid))
        else:
            if boundary_side == 'bottom':
                v_iso_lower.append(apex_vid)
            elif boundary_side == 'top':
                v_iso_upper.append(apex_vid)
            else:
                v_iso_interior.append((pos[1], apex_vid))

    # Sort interior apexes by parametric position
    u_iso_interior.sort(key=lambda x: x[0])
    v_iso_interior.sort(key=lambda x: x[0])
    print_if_verbose(f" !!!! the viso interior apex vids and their v positions are: {v_iso_interior}")
    print_if_verbose(f" !!!! the uiso interior apex vids and their u positions are: {u_iso_interior}")
    # Build merged edges for each periodic direction
    for sub_chains, lower_apexes, interior_apexes, upper_apexes in [
        (merged_u_sub_chains, v_iso_lower, v_iso_interior, v_iso_upper),
        (merged_v_sub_chains, u_iso_lower, u_iso_interior, u_iso_upper),
    ]:
        if len(sub_chains) == 0:
            continue

        interior_vids = [a[1] for a in interior_apexes]

        # Validate: number of interior apexes must equal number of gaps
        num_gaps = len(sub_chains) - 1
        if num_gaps > 0 and len(interior_vids) != num_gaps:
            raise ValueError(
                f"{len(sub_chains)} sub-chains require {num_gaps} interior apexes, "
                f"but got {len(interior_vids)}")

        # Build the full ordered sequence:
        # [lower_apex] → chain[0] → [interior_apex_0] → chain[1] → ... → [upper_apex]
        if lower_apexes:
            apex = lower_apexes[0]
            surface_merged_edges.append(create_edge(apex, sub_chains[0][0]))

        for k, chain in enumerate(sub_chains):
            for i in range(len(chain) - 1):
                surface_merged_edges.append(create_edge(chain[i], chain[i + 1]))
            if k < num_gaps:
                apex = interior_vids[k]
                surface_merged_edges.append(create_edge(chain[-1], apex))
                surface_merged_edges.append(create_edge(apex, sub_chains[k + 1][0]))

        if upper_apexes:
            apex = upper_apexes[0]
            surface_merged_edges.append(create_edge(sub_chains[-1][-1], apex))

    return surface_merged_edges


def split_between_two_stitching_vertices(V,
                                        connectivity,
                                        vid0,
                                        vid1,
                                        prohibited_vids: set = None):
    '''
    This function is called on two vertices that will be stitched together. 
    If the shortest path between the two vertices has fewer than four edges, we split the edges between them
    until it has at least four.
   
    Note, when calling, we assume vid0 and vid1 are matched vertices that will be stitched. 
    The screening is also done in this function.
    '''
    if vid0 == vid1:
        return

    traced_vids = bfs(connectivity.adj_list,
                      vid0,
                      vid1,
                      prohibited_vids)
    
    if not traced_vids:
        raise ValueError(f"Stitching two vertices but no path found between {vid0} and {vid1}")
    
    # we need at least four edges, so at least five vertices
    # .__.__.__.__.
    if len(traced_vids) > 4:
        return
    # split all the edges between vid0 and vid1
    while True:
        new_traced_vids = []
        for i in range(len(traced_vids)-1):
            v_start = traced_vids[i]
            v_end = traced_vids[i+1]
            
            split_edge_in_mesh_overloaded(V,
                                        connectivity,
                                        (v_start, v_end)    
                                        )
            if new_traced_vids:
                new_traced_vids.pop()  # remove the last one to avoid duplication
            new_traced_vids.append(v_start)
            new_traced_vids.append(len(V)-1)  # the newly inserted vertex
            new_traced_vids.append(v_end)
        if (len(new_traced_vids) > 4):
            traced_vids = new_traced_vids
            break
        traced_vids = new_traced_vids
    
    # This is not enough. 
    # We need to check again, if the two stitching vertices have edges that are directly connected.
    # For example, vid0 -- v_a -- vid1
    # We need to split vid0 -- v_a and vid1 -- v_a
    # Unless these edges are boundary edges. For boundary edges, merging will not cause non manifoldness.
    vo_neighbors = connectivity.adj_list[vid0]
    vi_neighbors = connectivity.adj_list[vid1]
    common_neighbors = vo_neighbors.intersection(vi_neighbors)
    for common_n in common_neighbors:
        if len(set(connectivity.VF[vid0]).intersection(connectivity.VF[common_n])) > 1:
            split_edge_in_mesh_overloaded(V,
                                          connectivity,
                                          (vid0, common_n)
                                         )
        if len(set(connectivity.VF[vid1]).intersection(connectivity.VF[common_n])) > 1:
            split_edge_in_mesh_overloaded(V,
                                      connectivity,
                                      (vid1, common_n)
                                     )

    return

def _split_edge_and_update_sing(V, connectivity, edge, sing_edges):
    """Split an edge and, if it appears in sing_edges, replace it with the two sub-edges."""
    split_edge_in_mesh_overloaded(V, connectivity, edge)
    if sing_edges is None:
        return
    new_vid = len(V) - 1
    edge_set = {edge[0], edge[1]}
    for idx, edges in sing_edges.items():
        for i, e in enumerate(edges):
            print_if_verbose(f"Checking if singularity edge {e} in index {idx} needs update after splitting edge {edge}")
            if {e[0], e[1]} == edge_set:
                edges[i] = (e[0], new_vid)
                edges.insert(i + 1, (new_vid, e[1]))
                print_if_verbose(f"Updated singularity edge {e} to {(e[0], new_vid)} and {(new_vid, e[1])} in singularity index {idx} after splitting edge {edge}")
                return

def split_between_two_stitching_vertices_updating_sing(V,
                                        connectivity,
                                        vid0,
                                        vid1,
                                        prohibited_vids: set = None,
                                        sing_edges: dict = None):
    '''
    Same as split_between_two_stitching_vertices but also updates sing_edges
    when a split affects a singularity edge.
    '''
    if vid0 == vid1:
        return

    traced_vids = bfs(connectivity.adj_list,
                      vid0,
                      vid1,
                      prohibited_vids)

    if not traced_vids:
        raise ValueError(f"Stitching two vertices but no path found between {vid0} and {vid1}")

    if len(traced_vids) > 4:
        return

    while True:
        new_traced_vids = []
        for i in range(len(traced_vids)-1):
            v_start = traced_vids[i]
            v_end = traced_vids[i+1]

            _split_edge_and_update_sing(V, connectivity, (v_start, v_end), sing_edges)

            if new_traced_vids:
                new_traced_vids.pop()
            new_traced_vids.append(v_start)
            new_traced_vids.append(len(V)-1)
            new_traced_vids.append(v_end)
        if (len(new_traced_vids) > 4):
            traced_vids = new_traced_vids
            break
        traced_vids = new_traced_vids
    vo_neighbors = connectivity.adj_list[vid0]
    vi_neighbors = connectivity.adj_list[vid1]
    common_neighbors = vo_neighbors.intersection(vi_neighbors)
    for common_n in common_neighbors:
        if len(set(connectivity.VF[vid0]).intersection(connectivity.VF[common_n])) > 1:
            _split_edge_and_update_sing(V, connectivity, (vid0, common_n), sing_edges)
        if len(set(connectivity.VF[vid1]).intersection(connectivity.VF[common_n])) > 1:
            _split_edge_and_update_sing(V, connectivity, (vid1, common_n), sing_edges)
    return