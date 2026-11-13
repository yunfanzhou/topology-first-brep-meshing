"""
Data structures for CAD mesh processing.

This module contains the core data structures used throughout the CAD mesh processing pipeline.
These structures provide a type-safe and organized way to handle mesh data, feature edges, loops, and patches.
"""
import numpy as np
from collections import defaultdict
from utils.print import print_if_verbose

class ConnectivityMaintainance:
    def __init__(self, 
                 F, 
                 VF:dict[list]= None, 
                 adj_list=None, 
                 edge_set_to_length_mapping=None,
                 adj_list_copy_dict=None,
                 modified_edge_set=None):
        self.F = F
        self.VF = VF or defaultdict(list)
        self.adj_list = adj_list or []
        self.edge_set_to_length_mapping = edge_set_to_length_mapping or {}
        self.adj_list_copy_dict = adj_list_copy_dict or {}
        self.modified_edge_set = modified_edge_set or set()
        self.F_len_before_tracing = len(F)
# ==========================================
# ==========================================
class PatchRegularSpaceData:
    def __init__(self,
                 prohibited_vids_set: set[int] = None,
                 frozen_edges_set : set[(int, int)] = None,
                 traced_edge_set : set[(int, int)] = None,
                 previous_loops: list[list[int]] = None, # can be empty while inserting the first loop
                 current_loop: list[int] = None
                 ):
        self.prohibited_vids_set = prohibited_vids_set or set()
        self.frozen_edges_set = frozen_edges_set or set() # includes boundary edges when tracing the initial loop, traced edges, topology recovery edges
        self.traced_edge_set = traced_edge_set or set() # overlap with the frozen edges set. 
                                                # only has traced edges, updated after each trace
        self.previous_loops = previous_loops or [] # can be empty while inserting the first loop
        self.current_loop = current_loop or []

class SingularityData:
    def __init__(self,
                 singularity_vids: list[int] = None,
                 position: np.array = None):
        self.singularity_vids = singularity_vids or []
        if position is not None and position.ndim == 1 and position.size == 3:
            self.position = position
        else:
            print_if_verbose(f"Warning: Invalid position: {position}")
            self.position = None
    def __repr__(self):
        return f"SingularityData(singularity_vids={self.singularity_vids}, position={self.position})"
class LoopGeometryData:
    def __init__(self, 
                 feature_id_poly_V_True_orientation: dict[int, np.ndarray] = None):
        self.feature_id_poly_V_True_orientation = feature_id_poly_V_True_orientation or {}
    def __repr__(self):
        return f"LoopGeometryData({self.feature_id_poly_V_True_orientation})"
    
class HalfEdge:
    def __init__(self, 
                 feature_id: int, 
                 orientation: bool,
                 is_degenerate: bool,
                 start_corner_id: int,
                 end_corner_id: int):
        self.feature_id = feature_id # feature id used to obtain the polyline V
        self.orientation = orientation 
        self.is_degenerate = is_degenerate # feature edge is degenerate when it's a singularity point
        self.start_corner_id = start_corner_id 
        self.end_corner_id = end_corner_id
    def __repr__(self):
        return f"HalfEdge(feature_id={self.feature_id}, orientation={self.orientation}, is_degenerate={self.is_degenerate}, start_corner_id={self.start_corner_id}, end_corner_id={self.end_corner_id})"
class LoopTopologyData:
    def __init__(self, 
                 halfedges: list[HalfEdge] = None,
                 seam_ids_set: set[int] = None):
        self.halfedges = halfedges or [] # a list of the halfedges (feature_id, orientation) that uniquely identifies a half-edge to HalfEdge object
                                               # the haledges are also in the order of each simple loop
        self.seam_ids_set = seam_ids_set or set() # set of edge feature_ids that are seams

    def __repr__(self):
        return f"LoopTopologyData(halfedges={self.halfedges}, seam_ids_set={self.seam_ids_set})"
class NonInitialLoopTreatmentData:
    def __init__(self,
                 already_treated_non_initial_loops: set[int] = None, # index of previous loops (not loopid) that already connected to the boundary)
                 loops_need_treatment: list[int] = None# index of previous loops (not loopid) that are treated in the previous steps
                 ):
        self.already_treated_non_initial_loops = already_treated_non_initial_loops or set()
        self.loops_need_treatment = loops_need_treatment or []

class CornerData:
    def __init__(self,
                 corner_id_to_position):
        self.corner_id_to_position = corner_id_to_position
class PatchInputData:
    def __init__(self,
                 V=None, 
                 F=None,
                 initial_loop_id=None,
                 patch_loop_geometry_data: dict[int, LoopGeometryData] = None,
                 patch_loop_topology_data: dict[int, LoopTopologyData] = None,
                 u_periodic_edges: tuple[list[int], list[int]] = None,
                 v_periodic_edges: tuple[list[int], list[int]] = None,
                 surface_merged_edges: list[(int, int)] = None):
        self.V = V if V is not None else []
        self.F = F if F is not None else []
        self.initial_loop_id = initial_loop_id
        self.patch_loop_geometry_data = patch_loop_geometry_data or {}
        self.patch_loop_topology_data = patch_loop_topology_data or {}
        self.u_periodic_edges = u_periodic_edges or ([], [])
        self.v_periodic_edges = v_periodic_edges or ([], [])
        self.surface_merged_edges = surface_merged_edges or []

    def __repr__(self):
        return (f"PatchInputData(V.size()={len(self.V)}, F.size()={len(self.F)}, initial_loop_id={self.initial_loop_id}, "
                f"patch_loop_geometry_data={self.patch_loop_geometry_data}, "
                f"patch_loop_topology_data={self.patch_loop_topology_data}, "
                f"u_periodic_edges={self.u_periodic_edges}, "
                f"v_periodic_edges={self.v_periodic_edges}, "
                f"surface_merged_edges={self.surface_merged_edges})")

class EdgeTracingData:
    def __init__(self,
                 traced_vids : list[int] = None,
                 segment_upsample_size : list[int] = None,
                 start_vid : int = None,
                 end_vid: int = None):
        self.traced_vids = traced_vids or []
        self.segment_upsample_size = segment_upsample_size or []
        self.start_vid = start_vid
        self.end_vid = end_vid
    def __repr__(self):
        return (f"EdgeTracingData(traced_vids={self.traced_vids}, "
                f"segment_upsample_size={self.segment_upsample_size}, "
                f"start_vid={self.start_vid}, end_vid={self.end_vid})")
class LoopTracingData:
    def __init__(self,
                feature_id_to_edge_tracing_data : dict[int, list[int]] = None,
                feature_id_to_segment_upsample_size : dict[int, list[int]] = None,
                feature_id_to_edge_orientation : dict[int, bool] = None,
                corner_id_to_vid : dict[int, int] = None,
                traced_id_set: set[int] = None):
        self.feature_id_to_edge_tracing_data = feature_id_to_edge_tracing_data or {}
        self.feature_id_to_segment_upsample_size = feature_id_to_segment_upsample_size or {}
        self.feature_id_to_edge_orientation = feature_id_to_edge_orientation or {} # the orientation of the edge that is traced
        self.corner_id_to_vid = corner_id_to_vid or dict() # corner id to vid
        self.traced_id_set = traced_id_set or set() # this set is used to check if a merged edge (seam edge) has been traced
    def __repr__(self):
        return (f"LoopTracingData(feature_id_to_edge_tracing_data.keys()={self.feature_id_to_edge_tracing_data.keys()}, "
                f"feature_id_to_segment_upsample_size.keys()={self.feature_id_to_segment_upsample_size.keys()}, "
                f"feature_id_to_edge_orientation={self.feature_id_to_edge_orientation}, "
                f"corner_id_to_vid={self.corner_id_to_vid}, "
                f"traced_id_set={self.traced_id_set})")
class InitialLoopTracingData:
    def __init__(self,
                 initial_loop_id: int = None,
                 loop_tracing_data: LoopTracingData = None,
                 singularity_corner_mapping: dict[int, int] = None):
        ''' The singularity_corner_mapping maps corner ids to apex vertex ids.
        It is the same as corner_id_to_vid in the LoopTracingData. 
        The only difference is that it will be preserved upon rollback, while corner_id_to_vid will be reset. 
        So we treat singularity corners differently from regular corners.
        '''    
        self.initial_loop_id = initial_loop_id
        self.loop_tracing_data = loop_tracing_data or LoopTracingData()
        self.singularity_corner_mapping = singularity_corner_mapping or {} # corner id to apex vid
    def __repr__(self):
        return (f"InitialLoopTracingData(initial_loop_id={self.initial_loop_id}, "
                f"loop_tracing_data={self.loop_tracing_data}, "
                f"singularity_corner_mapping={self.singularity_corner_mapping})")
class FallBackLoopTracingData:
    def __init__(self,
                 halfedges_tracing_data : list[EdgeTracingData] = None,
                 halfcorner_vids : list[int] = None):
        self.halfedges_tracing_data = halfedges_tracing_data or []
        self.halfcorner_vids = halfcorner_vids or []
class PatchTracingData:
    def __init__(self,
                 initial_loop_tracing_data : InitialLoopTracingData = None,
                 remaining_loops_tracing_data : dict[int, LoopTracingData] = None):
        self.initial_loop_tracing_data = initial_loop_tracing_data or InitialLoopTracingData()
        self.remaining_loops_tracing_data = remaining_loops_tracing_data or {}
    def __repr__(self):
        return (f"PatchTracingData(initial_loop_tracing_data={self.initial_loop_tracing_data}, "
                f"remaining_loops_tracing_data={self.remaining_loops_tracing_data})")
class AggregatedPatchTracingData:
    def __init__(self,
                 feature_id_to_traced_vids : dict[int, list[int]] = None,
                 feature_id_to_segment_upsample_size : dict[int, list[int]] = None,
                 feature_id_to_orientation : dict[int, bool] = None,
                 corner_id_to_vid : dict[int, int] = None):
        self.feature_id_to_traced_vids = feature_id_to_traced_vids or {}
        self.feature_id_to_segment_upsample_size = feature_id_to_segment_upsample_size or {}
        self.feature_id_to_orientation = feature_id_to_orientation or {}
        self.corner_id_to_vid = corner_id_to_vid or {}
    def __repr__(self):
        return (f"AggregatedPatchTracingData(feature_id_to_traced_vids.keys()={self.feature_id_to_traced_vids.keys()}, "
                f"feature_id_to_segment_upsample_size.keys()={self.feature_id_to_segment_upsample_size.keys()}, "
                f"feature_id_to_orientation={self.feature_id_to_orientation}, "
                f"corner_id_to_vid={self.corner_id_to_vid})")