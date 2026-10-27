# config.py

verbose = 0

# --- Oversized-patch guard ---------------------------------------------------
# When a patch's vertex count crosses ``oversized_vertex_threshold`` during the
# local step (runaway regular-space refinement), the mesh state is dumped so it
# can be investigated before an OOM kill.  See
# ``utils.print.dump_oversized_patch_state``.
oversized_vertex_threshold = 1_000_000
# Relative to the working directory by default; main.py points it at
# <output_folder>/oversized_patch_dumps (in worker processes too).
oversized_dump_dir = "oversized_patch_dumps"
# Set by ``local_step_one_patch`` so the dump is labelled with the patch id.
current_patch_id = None

# --- Posterior singularity detection ----------------------------------------
# After the first mmg pass of the base mesh, and only when the exporter gave
# the face no singularity records, each domain-boundary iso-line's lifted
# vertex chain is measured in 3-D.  A chain shorter than
# ``collapsed_iso_line_rel_tol`` times the model bbox diagonal is a collapsed
# (apex) line: a singularity record is synthesized for it and the face is
# meshed once more through the regular singularity path.  Apex lines sit at
# kernel tolerance (~1e-9 of the model), real edges at >= 1e-4.  A collapsed
# iso-line strictly inside the domain is never synthesized: it raises
# ``initial_meshing.CollapsedInteriorIsoLineError`` and ends the run.
collapsed_iso_line_rel_tol = 1e-7
# Bound on the manifold retry loop in run_initial_meshing_for_face.
manifold_refine_max = 4
