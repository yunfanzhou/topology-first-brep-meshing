import numpy as np
import igl
import copy 
from utils.mesh_utils import update_edge_length_and_adjacency_list, update_adjacency_list, create_edge, compute_VF
from utils.print import print_if_verbose

def split_edge_in_mesh_overloaded(V, connectivity, edge, new_point = None, faces = None):
    return split_edge_in_mesh(V, connectivity.F, edge, connectivity.VF, new_point, faces, connectivity.adj_list, connectivity.edge_set_to_length_mapping, connectivity.adj_list_copy_dict, connectivity.modified_edge_set)

def split_edge_in_mesh(V, 
                       F, 
                       edge, 
                       VF, 
                       new_point= None, 
                       faces=None, 
                       adjacency_list=None, 
                       edge_mapping_to_length = None, 
                       adj_list_copy_dict = None, 
                       modified_edge_set = None):
    # Split an edge in the mesh
    # Input:
    # V: vertices of the mesh
    # F: faces of the mesh
    # edge: the edge to split
    # VF: vertex-to-faces map; the faces to split are the ones shared by both edge vertices
    # new_point: position of the new vertex (default: the edge midpoint)
    # Output (V, F, VF and the connectivity structures are modified in place):
    # V: V after splitting the edge
    # F: F after splitting the edge
    # adjacency_list, edge_mapping_to_length: the updated structures
    common_faces = set(VF[edge[0]]) & set(VF[edge[1]])
    faces = list(common_faces)

    if new_point is None:
        new_point  = (np.array(V[edge[0]]) + np.array(V[edge[1]])) / 2
    newv_index = len(V)
    
    V.append(new_point)
    VF[newv_index] = [f for f in faces]
    edge_adjacent_faces = []
    for f in faces:
        if f == -1:
            continue

        face = F[f]
        old_face = [vid for vid in face]
        edge_adjacent_faces.append(old_face)
        idx = {v: i for i, v in enumerate(face)}

        if edge[0] not in idx or edge[1] not in idx:
            print_if_verbose(f"    Error: edge {edge} not found in face {face}")
            continue

        index0, index1 = idx[edge[0]], idx[edge[1]]

        new_face = [vid for vid in F[f]]
        new_face[index1] = newv_index

        F.append(new_face)  # list append is very fast

        F[f][index0] = newv_index

        newf = len(F) - 1  # new face index

        VF[newv_index].append(newf)
        VF[edge[0]].remove(f)
        VF[edge[0]].append(newf)

        index2 = ({0, 1, 2} - {index0, index1}).pop()
        VF[new_face[index2]].append(newf)
    # after everything
    if adjacency_list is not None and edge_mapping_to_length is not None:
        adjacency_list, edge_mapping_to_length = update_edge_length_and_adjacency_list(V, 
                                                                                       newv_index, 
                                                                                       edge, 
                                                                                       edge_adjacent_faces, 
                                                                                       adjacency_list, 
                                                                                       edge_mapping_to_length, 
                                                                                       adj_list_copy_dict, 
                                                                                       modified_edge_set)
    elif adjacency_list is not None:
        adjacency_list = update_adjacency_list( newv_index, edge, edge_adjacent_faces, adjacency_list)
    return V, F, adjacency_list, edge_mapping_to_length

def compute_bary(V, face, point):
    v0 = np.array(V[face[0]])
    v1 = np.array(V[face[1]])
    v2 = np.array(V[face[2]])
    T = np.stack([v0 - v2, v1 - v2], axis=1)
    p = np.array(point) - v2
    lam, _, _, _ = np.linalg.lstsq(T, p, rcond=None)
    return np.array([lam[0], lam[1], 1.0 - lam[0] - lam[1]])

def point_triangle_sqr_dist(V, face, point):
    # same as in igl/squared_distance_to_triangle.cpp
    a = np.array(V[face[0]])
    b = np.array(V[face[1]])
    c = np.array(V[face[2]])
    p = np.array(point)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return np.dot(ap, ap)
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return np.dot(bp, bp)
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return np.dot(cp, cp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        closest = a + v * ab
        diff = p - closest
        return np.dot(diff, diff)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        closest = a + w * ac
        diff = p - closest
        return np.dot(diff, diff)
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest = b + w * (c - b)
        diff = p - closest
        return np.dot(diff, diff)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    closest = a + v * ab + w * ac
    diff = p - closest
    return np.dot(diff, diff)

def embed_point_in_triangle(V, connectivity, point, frozen_edges_set, prohibited_vids_set,
                             closest_fid=None, candidate_fids=None,
                            off_boundary=True, *, prev_vid, traced_edge_set):
    # The point never snaps onto an already traced vertex, whatever off_boundary
    # says: prev_vid (the previous sample's vertex, None for an edge's first sample;
    # not on a traced edge yet before its first segment is traced) or any vertex of
    # traced_edge_set. Snapping there would give a zero-length segment or merge two
    # traced curves; the point is inserted by an edge or face split instead. Both
    # are required so no caller can skip the rule. (Traced corners are reused
    # through the corner mapping, never through this embedding.)
    def is_traced_vertex(v):
        return v == prev_vid or any(create_edge(v, w) in traced_edge_set
                                    for w in connectivity.adj_list[v])

    BARY_TOL = 0.1
    EDGE_T_TOL = 1e-2
    # "Inside" tolerance for the point's projection: samples lie on the CAD surface while the mesh
    # only approximates it, so near edges/vertices the projection can fall slightly outside the
    # closest face (b ~ -0.01). Wildly extrapolated barycentrics (b ~ -2 .. -100, point far from
    # the face) must still be rejected, so the tolerance is of the same order as BARY_TOL.
    BARY_INSIDE_TOL = BARY_TOL
    CONTAIN_CHECK_K = 16
    all_candidate_fids = []
    if candidate_fids is not None:
        all_candidate_fids.extend(candidate_fids)
    elif closest_fid is not None:
        all_candidate_fids.append(closest_fid)
    new_fids = range(connectivity.F_len_before_tracing, len(connectivity.F))
    all_candidate_fids.extend(new_fids)

    def _dist_pairs(fids):
        pairs = [(int(fid), point_triangle_sqr_dist(V, connectivity.F[fid], point)) for fid in fids]
        pairs.sort(key=lambda x: x[1])  # stable
        return pairs

    def _projects_inside(fid):
        return bool(np.all(compute_bary(V, connectivity.F[fid], point) >= -BARY_INSIDE_TOL))

    def _find_host(pairs):
        # A host face must contain the point's projection AND be (within rounding) as close as the
        # closest face we know of. compute_bary alone is a plane projection: a far face can
        # "contain" the projection of a point that actually lies on/near a different face.
        if not pairs:
            return None
        best_d = pairs[0][1]
        limit = best_d * (1.0 + 1e-6) + 1e-12
        for fid, d in pairs:
            if d > limit:
                break
            if _projects_inside(fid):
                return fid
        return None

    pairs = _dist_pairs(dict.fromkeys(int(f) for f in all_candidate_fids))
    sorted_fids = [fid for fid, _ in pairs]

    # The k-NN candidates come from a barycenter kd-tree, which can miss a large triangle that
    # actually contains the point (its barycenter is far away while small faces nearby are not).
    # Look for a host among the closest candidates, then their one-ring (cheap), and only as a
    # last resort run a full-mesh AABB query (O(#F) per call; too slow to run for every sample).
    head = pairs[:CONTAIN_CHECK_K]
    host = _find_host(head)
    if host is None:
        ring = set()
        for fid, _ in head:
            for v in connectivity.F[fid]:
                ring.update(int(f) for f in connectivity.VF[v])
        ring.difference_update(sorted_fids)
        host = _find_host(_dist_pairs(ring) + head)
    if host is None:
        V_np = np.array(V)
        F_np = np.array(connectivity.F)
        _, I, _ = igl.point_mesh_squared_distance(np.array([point]), V_np, F_np)
        host = int(np.atleast_1d(I)[0])  # truly closest face; the centroid fallback below uses it
    sorted_fids = [host] + [fid for fid in sorted_fids if fid != host]

    for closest_fid in sorted_fids:
        closest_face = connectivity.F[closest_fid]
        b = compute_bary(V, closest_face, point)
        # Snapping heuristics are only meaningful when the point projects inside this face;
        # extrapolated barycentrics (point outside the face) must never trigger a snap.
        inside = bool(np.all(b >= -BARY_INSIDE_TOL))
        near_vertex = inside & (b > (1.0 - BARY_TOL))
        near_edge   = inside & (b < BARY_TOL)
        near_vertex &= np.array([not is_traced_vertex(v) for v in closest_face])

        if off_boundary:
            for i in range(3):
                if near_vertex[i]:
                    vi = closest_face[i]
                    if vi in prohibited_vids_set:
                        near_vertex[i] = False
                if near_edge[i]:
                    va = closest_face[(i + 1) % 3]
                    vb = closest_face[(i + 2) % 3]
                    e = create_edge(va, vb)
                    if e in frozen_edges_set:
                        near_edge[i] = False

        if np.any(near_vertex):
            masked_b = np.where(near_vertex, b, -1.0)
            vert_idx = int(np.argmax(masked_b))
            embedded_vid = closest_face[vert_idx]
            return embedded_vid, closest_face

        if np.any(near_edge):
            masked_b_edge = np.where(near_edge, b, np.inf)
            edge_opp_idx = int(np.argmin(masked_b_edge))
            va = closest_face[(edge_opp_idx + 1) % 3]
            vb = closest_face[(edge_opp_idx + 2) % 3]
            b_va = b[(edge_opp_idx + 1) % 3]
            b_vb = b[(edge_opp_idx + 2) % 3]
            t = b_vb / (b_va + b_vb)
            t = np.clip(t, 0.0, 1.0)
            if t < EDGE_T_TOL or t > 1.0 - EDGE_T_TOL:
                continue
            split_point = (1.0 - t) * np.array(V[va]) + t * np.array(V[vb])
            split_edge_in_mesh_overloaded(V, connectivity, [va, vb], new_point=split_point)
            embedded_vid = len(V) - 1
            return embedded_vid, closest_face

        embedded_vid, closest_face = split_1_to_3(V, connectivity, point, closest_fid)
        return embedded_vid, closest_face

    closest_fid = sorted_fids[0]
    closest_face = connectivity.F[closest_fid]
    embedded_vid, closest_face = split_1_to_3(V, connectivity, point, closest_fid)
    return embedded_vid, closest_face


def split_1_to_3(V, connectivity, point, closest_fid=None):
    # Insert a vertex at the centroid of a face, splitting it 1-to-3
    # Input:
    # V: vertices of the mesh
    # connectivity: mesh connectivity (F, VF, adj_list, edge lengths), updated in place
    # point: only used to find the closest face when closest_fid is None
    # closest_fid: the face to split
    # Output:
    # inserted_vid: the vertex id that is inserted and traced
    # closest_face: the vertices of the face that was split
    if closest_fid is None:
        V_np = np.array(V)
        F_np = np.array(connectivity.F)
        sqrD, I, C = igl.point_mesh_squared_distance(np.array([point]), V_np, F_np)
        closest_fid = I.tolist()
        closest_face = F_np[closest_fid]
        new_point = (V_np[closest_face[0]] + V_np[closest_face[1]] + V_np[closest_face[2]]) / 3. # center of the closest face
    else :
        closest_face = connectivity.F[closest_fid]
        new_point = (np.array(V[closest_face[0]]) + np.array(V[closest_face[1]]) + np.array(V[closest_face[2]])) / 3.
    # TODO We optimize the point to be the closest corner of the face if the triangle is not boundary triangle 

    newV_index = len(V)

    V.append(new_point)

    connectivity.F.append([newV_index, closest_face[1], closest_face[2]])
    connectivity.F.append([closest_face[0], newV_index, closest_face[2]])
    connectivity.F[closest_fid] = [closest_face[0], closest_face[1], newV_index]

    F_size = len(connectivity.F)

    connectivity.VF[newV_index] = [F_size - 2, F_size - 1, closest_fid]
    connectivity.VF[closest_face[0]].append(F_size - 1)
    connectivity.VF[closest_face[1]].append(F_size - 2)
    connectivity.VF[closest_face[2]].remove(closest_fid)
    connectivity.VF[closest_face[2]].append(F_size - 1)
    connectivity.VF[closest_face[2]].append(F_size - 2)

    update_connectivity_after_1_to_3_split(V, connectivity, newV_index, closest_face)
    return newV_index, closest_face
def update_connectivity_after_1_to_3_split(V, 
                                           connectivity,
                                           newv_index, 
                                           old_face):
    # Ensure the new vertex's adjacency set contains all three old face vertices
    connectivity.adj_list.append(set(old_face))
    for i in range(3):
        vertex = old_face[i]
        if connectivity.adj_list_copy_dict is not None and vertex not in connectivity.adj_list_copy_dict:
            connectivity.adj_list_copy_dict[vertex] = copy.deepcopy(connectivity.adj_list[vertex])
        connectivity.adj_list[vertex].add(newv_index)
        connectivity.adj_list[newv_index].add(vertex)  # <-- Ensure symmetry
        edge = create_edge(vertex, newv_index)
        if connectivity.modified_edge_set is not None:
            connectivity.modified_edge_set.add(edge)
        connectivity.edge_set_to_length_mapping[edge] = np.linalg.norm(np.array(V[vertex]) - np.array(V[newv_index]))

    return

def collapse_nonmanifold_prevention(V,
                                    connectivity, 
                                    collapsing_edge,
                                    frozen_edge_set = None):
    ''' checking the link condition.'''
    vertices_breaking_link_condition = edge_link_condition(connectivity, collapsing_edge)
    if len(vertices_breaking_link_condition) == 0:
        return
    for v in vertices_breaking_link_condition:
        for vid in collapsing_edge:
            edge = create_edge(v, vid)
            if frozen_edge_set is not None and edge in frozen_edge_set:
                continue
            else:
                split_edge_in_mesh_overloaded(V,
                                              connectivity,
                                              edge)

    # check link condition once more 
    vertices_breaking_link_condition = edge_link_condition(connectivity, collapsing_edge)
    if len(vertices_breaking_link_condition) != 0:
        print_if_verbose(f"  After splitting edges, collapsing edge {collapsing_edge} still violates link condition.")
        raise ValueError(f"After splitting edges, collapsing edge {collapsing_edge} still violates link condition.")

def edge_link_condition(connectivity, collapsing_edge):
    # find the edges that need to be split to prevent non-manifold after collapsing the collapsing_edge
    v0 = collapsing_edge[0]
    v1 = collapsing_edge[1]
    v0_neighbors = connectivity.adj_list[v0]
    v1_neighbors = connectivity.adj_list[v1]
    vertices_common_neighbors = set(v0_neighbors) & set(v1_neighbors)
    #link for the edge v0-v1
    edge_link_vids = set()
    for f in set(connectivity.VF[v0]) & set(connectivity.VF[v1]):
        face = connectivity.F[f]
        for i in range(3):
            if face[i] != v0 and face[i] != v1:
                edge_link_vids.add(face[i])
                break
    
    if vertices_common_neighbors != edge_link_vids:
        print_if_verbose(f"  Collapsing: edge {collapsing_edge} violates link condition. Splitting necessary edges.")
        return vertices_common_neighbors - edge_link_vids
    return set()
def safe_collapse_with_frozen_edges(V, 
                                   connectivity, 
                                   collapsing_edge : list[int], 
                                   collapse_to_position,
                                   collapse_to_vid = None,
                                   frozen_edge_set = None):
    # Collapse an edge in the mesh
    # If the edge fails the link condition, we first split the edges that break it.
    # However, this might not always work, since edges that break the link condition can be frozen edges.

    # Triangles that contain both vertices of the collapsing edge go to the will_delete set;
    # the ones that contain only one of them go to the will_edit set.
    collapse_nonmanifold_prevention(V, 
                                    connectivity, 
                                    collapsing_edge,
                                    frozen_edge_set)
    collapsing_vid_set = set([collapsing_edge[0], collapsing_edge[1]])
    will_delete = set()
    will_edit = set()

    for i in range(len(connectivity.F)):
        triangle = connectivity.F[i]
        if len(set(triangle) & collapsing_vid_set) == 2:
            will_delete.add(i)
        elif len(set(triangle) & collapsing_vid_set) == 1:
            will_edit.add(i)

    if len(will_delete) == 0 and len(will_edit) == 0:
        print_if_verbose("No triangles to edit or delete for forming cone. Returning None.")
        return None

    # edit the triangles that contain the collapsing vids
    if collapse_to_vid is None:
        collapse_to_vid = collapsing_edge[0]
    collapsing_vid_set.remove(collapse_to_vid)
    V[collapse_to_vid] = collapse_to_position
    for i in will_delete:
        triangle = connectivity.F[i]
        for j in range(3):
            connectivity.adj_list_copy_dict[triangle[j]] = [f for f in connectivity.adj_list[triangle[j]]]
            connectivity.modified_edge_set.add(create_edge(triangle[j], triangle[(j + 1) % 3]))
    for i in will_edit:
        triangle = connectivity.F[i]
        for j in range(3):
            if triangle[j] not in collapsing_vid_set:
                continue
            vid_to_remove = triangle[j]
            connectivity.modified_edge_set.add(create_edge(vid_to_remove, triangle[(j + 1) % 3]))
            connectivity.modified_edge_set.add(create_edge(vid_to_remove, triangle[(j + 2) % 3]))
            
            triangle[j] = collapse_to_vid
            # remove the edge from the connectivity.edge_set_to_length_mapping
            connectivity.edge_set_to_length_mapping.pop(create_edge(vid_to_remove, triangle[(j + 1) % 3]), None)
            connectivity.edge_set_to_length_mapping.pop(create_edge(vid_to_remove, triangle[(j + 2) % 3]), None)
            connectivity.edge_set_to_length_mapping[create_edge(collapse_to_vid, triangle[(j + 1) % 3])] = np.linalg.norm(np.array(V[collapse_to_vid]) - np.array(V[triangle[(j + 1) % 3]]))
            connectivity.edge_set_to_length_mapping[create_edge(collapse_to_vid, triangle[(j + 2) % 3])] = np.linalg.norm(np.array(V[collapse_to_vid]) - np.array(V[triangle[(j + 2) % 3]]))
            # add the new edge created to the connectivity.modified_edge_set
            connectivity.modified_edge_set.add(create_edge(collapse_to_vid, triangle[(j + 1) % 3]))
            connectivity.modified_edge_set.add(create_edge(collapse_to_vid, triangle[(j + 2) % 3]))

            # Remove vid_to_remove from all its neighbors' adjacency sets
            for neighbor in list(connectivity.adj_list[vid_to_remove]):
                if neighbor not in connectivity.adj_list_copy_dict:
                    connectivity.adj_list_copy_dict[neighbor] = [adj for adj in connectivity.adj_list[neighbor]]
                connectivity.adj_list[neighbor].discard(vid_to_remove)
            # Clear vid_to_remove's adjacency set
            if vid_to_remove not in connectivity.adj_list_copy_dict:
                connectivity.adj_list_copy_dict[vid_to_remove] = [adj for adj in connectivity.adj_list[vid_to_remove]]
            connectivity.adj_list[vid_to_remove] = set()

            # Add collapse_to_vid to the neighbors' adjacency sets and vice versa
            for neighbor in [triangle[(j + 1) % 3], triangle[(j + 2) % 3]]:
                if collapse_to_vid not in connectivity.adj_list_copy_dict:
                    connectivity.adj_list_copy_dict[collapse_to_vid] = [adj for adj in connectivity.adj_list[collapse_to_vid]]
                if neighbor not in connectivity.adj_list_copy_dict:
                    connectivity.adj_list_copy_dict[neighbor] = [adj for adj in connectivity.adj_list[neighbor]]
                connectivity.adj_list[collapse_to_vid].add(neighbor)
                connectivity.adj_list[neighbor].add(collapse_to_vid)
                # Remove vid_to_remove from neighbor's adjacency set (safe to repeat)
                connectivity.adj_list[neighbor].discard(vid_to_remove)
            break
            

    # delete the triangles that in the will_delete set
    newF = [connectivity.F[i] for i in range(len(connectivity.F)) if i not in will_delete]
    connectivity.F = newF
    connectivity.VF = compute_VF(connectivity.F)
    connectivity.modified_edge_set = set()
    connectivity.adj_list_copy_dict = {}


    if igl.is_edge_manifold(np.array(connectivity.F)) == False:
        print_if_verbose("After collapsing, the mesh is not edge manifold.")
        raise Exception("After collapsing, the mesh is not edge manifold.")
    return collapse_to_vid