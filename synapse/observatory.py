"""Schematic source/role graph layout and actual executing activity.

Coordinates are generated display positions, never measured anatomy. Catalog
metadata identifies a source; executing provenance identifies the representation
actually loaded. Base adjacency and active graft links remain distinguishable.
"""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from .agent import IntegratedAgent
from .connectome_datasets import CONNECTOME_CATALOG, ALL_DATASET_IDS


class NeuralObservatory:
    """Generates reproducible display positions and runtime telemetry."""

    # Kept for API compatibility; actual display centers follow configured sources.
    NEUROPIL_OFFSETS: Dict[int, np.ndarray] = {
        0: np.array([0.0, 0.4, 0.0], dtype=np.float32),    # Region 0: Central Brain / FlyWire / MaleCNS
        1: np.array([0.0, -0.6, 0.0], dtype=np.float32),   # Region 1: MANC / VNC Cord / BANC
        2: np.array([-0.8, 0.3, 0.2], dtype=np.float32),   # Region 2: Optic Lobe (left anterior)
        3: np.array([0.3, -0.6, -0.2], dtype=np.float32),  # Region 3: FANC (lower right)
        4: np.array([0.0, 0.7, 0.1], dtype=np.float32),    # Region 4: FlyWire FAFB
        5: np.array([-0.3, -0.6, 0.2], dtype=np.float32),  # Region 5: MANC
        6: np.array([0.6, 0.1, 0.3], dtype=np.float32),    # Region 6: Larval Brain
        7: np.array([0.5, 0.4, -0.2], dtype=np.float32),   # Region 7: Hemibrain
    }

    NEUROPIL_NAMES: Dict[int, str] = {
        0: 'Central Brain / FlyWire / MaleCNS',
        1: 'VNC Nerve Cord / MANC / BANC',
        2: 'FlyEM Optic Lobe',
        3: 'FANC Female Nerve Cord',
        4: 'FlyWire FAFB v783',
        5: 'MANC Male Nerve Cord',
        6: 'Larval Drosophila Brain',
        7: 'Hemibrain Central Complex',
    }

    def __init__(self, agent: IntegratedAgent, seed: int = 42):
        self.agent = agent
        self.rng = np.random.default_rng(seed)
        self.source_names = tuple(getattr(agent.config, 'connectome_sources', ('flywire', 'manc', 'optic-lobe')))
        count = len(self.source_names)
        self.source_centers = {i: np.array([.68 * math.cos(i / count * 2 * math.pi - math.pi / 2),
                                           .68 * math.sin(i / count * 2 * math.pi - math.pi / 2), 0.], dtype=np.float32)
                               if count > 1 else np.zeros(3, dtype=np.float32) for i in range(count)}
        self._node_positions_3d: Optional[np.ndarray] = None

    @property
    def node_positions_3d(self) -> np.ndarray:
        if self._node_positions_3d is None or len(self._node_positions_3d) != self.agent.brain.n:
            self._node_positions_3d = self._compute_3d_positions()
        return self._node_positions_3d

    @node_positions_3d.setter
    def node_positions_3d(self, value: Optional[np.ndarray]) -> None:
        self._node_positions_3d = value

    def _compute_3d_positions(self) -> np.ndarray:
        """Assign stable schematic coordinates grouped by source and role."""
        n = self.agent.brain.n
        pos = np.zeros((n, 3), dtype=np.float32)
        if n == 0:
            return pos

        graph = self.agent.graph
        origins = graph.origin
        roles = graph.role
        n_orig = len(origins)
        n_roles = len(roles)
        # Vectorized center lookup + role z-offset + single PCG64 normal draw
        # (bit-identical RNG stream to n calls of rng.normal(..., size=3)).
        max_center = max(self.source_centers.keys(), default=0) + 1
        center_table = np.zeros((max_center, 3), dtype=np.float32)
        for idx_c, vec in self.source_centers.items():
            center_table[idx_c] = vec
        if n_orig >= n:
            orig_arr = np.asarray(origins[:n], dtype=np.int64)
        else:
            orig_arr = np.zeros(n, dtype=np.int64)
            if n_orig > 0:
                orig_arr[:n_orig] = origins
        valid_orig = (orig_arr >= 0) & (orig_arr < max_center)
        pos[valid_orig] = center_table[orig_arr[valid_orig]]

        role_z = {'sensory': -0.25, 'visual': -0.17, 'ascending': -0.08,
                  'central': 0.0, 'descending': 0.14, 'motor': 0.25}
        role_list = roles[:n].tolist() if hasattr(roles, 'tolist') else list(roles[:n])
        if n_roles < n:
            role_list.extend(['central'] * (n - n_roles))
        pos[:, 2] += np.fromiter((role_z.get(str(r), 0.0) for r in role_list), dtype=np.float32, count=n)
        pos += self.rng.normal(0.0, 0.10, size=(n, 3)).astype(np.float32)
        return pos

    def get_structure_payload(self) -> Dict[str, Any]:
        """Returns 3D nodes and edges for 3D visualizer canvas or WebGL sidecar."""
        n = self.agent.brain.n
        source_names = self.source_names
        provenance = getattr(self.agent, 'source_provenance', [])
        graph = self.agent.graph
        positions = self.node_positions_3d

        # Pre-resolve per-source catalog metadata and provenance templates once
        n_sources = len(source_names)
        source_meta = [CONNECTOME_CATALOG.get(s_name) for s_name in source_names]
        source_ds_name = [m.name if m else s_name for s_name, m in zip(source_names, source_meta)]
        source_license = [m.license if m else 'unknown' for m in source_meta]
        source_authors = [m.organisation_authors if m else 'unknown' for m in source_meta]
        prov_templates = [deepcopy(provenance[i]) if i < len(provenance) else {'representation': 'unknown'}
                          for i in range(max(n_sources, len(provenance)))]

        origins_list = graph.origin[:n].tolist() if hasattr(graph.origin, 'tolist') else [int(x) for x in graph.origin[:n]]
        if len(origins_list) < n:
            origins_list.extend([0] * (n - len(origins_list)))
        roles_list = [str(x) for x in (graph.role[:n].tolist() if hasattr(graph.role, 'tolist') else graph.role[:n])]
        if len(roles_list) < n:
            roles_list.extend(['central'] * (n - len(roles_list)))
        cell_types_list = [str(x) for x in (graph.cell_type[:n].tolist() if hasattr(graph.cell_type, 'tolist') else graph.cell_type[:n])]
        if len(cell_types_list) < n:
            cell_types_list.extend(['interneuron'] * (n - len(cell_types_list)))

        pos_rounded = np.round(positions.astype(np.float64), 4).tolist()
        nodes = []
        # Simultaneously group indices by (origin, role) in one O(N) pass for LOD clusters
        by_origin_role: Dict[Tuple[int, str], List[int]] = {}
        for i in range(n):
            origin = int(origins_list[i])
            role = roles_list[i]
            cell_type = cell_types_list[i]
            px, py, pz = pos_rounded[i]
            if 0 <= origin < n_sources:
                dataset_id = source_names[origin]
                ds_name = source_ds_name[origin]
                lic = source_license[origin]
                auth = source_authors[origin]
            else:
                dataset_id = 'unknown'
                ds_name = 'unknown'
                lic = 'unknown'
                auth = 'unknown'
            prov = dict(prov_templates[origin]) if 0 <= origin < len(prov_templates) else {'representation': 'unknown'}

            nodes.append({
                'id': i,
                'name': f"N{i}_{cell_type[:3]}",
                'x': px,
                'y': py,
                'z': pz,
                'neuropil': origin,
                'dataset': dataset_id,
                'dataset_name': ds_name,
                'license': lic,
                'authors': auth,
                'role': role,
                'cell_type': cell_type,
                'provenance': prov,
            })
            if role in ('sensory', 'central', 'motor'):
                by_origin_role.setdefault((origin, role), []).append(i)

        # Export bounded samples separately, with actual active graft membership.
        base_count = len(graph.base_pre)
        edge_idx = np.linspace(0, base_count - 1, min(2000, base_count), dtype=int) if base_count else np.empty(0, dtype=int)
        edges = [
            {'source': s, 'target': t, 'weight': round(w, 4), 'is_graft': False}
            for s, t, w in zip(
                graph.base_pre[edge_idx].tolist(),
                graph.base_post[edge_idx].tolist(),
                graph.base_w[edge_idx].tolist(),
            )
        ] if edge_idx.size else []
        g_pre = np.asarray(graph.graft_pre)
        g_post = np.asarray(graph.graft_post)
        g_w = self.agent.brain.graft_w.detach().cpu().numpy()

        active = self.agent.plasticity.active_mask.detach().cpu().numpy().astype(bool, copy=False)
        active_graft_count = int(np.count_nonzero(active))
        valid_rows = np.flatnonzero((g_pre < n) & (g_post < n) & active)[:2000]
        graft_links = [
            {'source': s, 'target': t, 'weight': round(w, 4), 'is_graft': True}
            for s, t, w in zip(
                g_pre[valid_rows].tolist(),
                g_post[valid_rows].tolist(),
                g_w[valid_rows].tolist(),
            )
        ] if valid_rows.size else []
        edges.extend({**link} for link in graft_links)

        neuropils_list = []
        for i, s_name in enumerate(source_names):
            neuropils_list.append({
                'id': i,
                'dataset_id': s_name,
                'name': source_ds_name[i],
                'authors': source_authors[i],
                'license': source_license[i],
                'center': self.source_centers[i].tolist(),
                'provenance': deepcopy(provenance[i]) if i < len(provenance) else {'representation': 'unknown'},
            })

        # Generate hierarchical Level-Of-Detail (LOD) clusters for scalable WebGL rendering
        lod_clusters = []
        cluster_id = 0
        for origin_idx, s_name in enumerate(source_names):
            for role_name in ('sensory', 'central', 'motor'):
                role_nodes = by_origin_role.get((origin_idx, role_name))
                if not role_nodes:
                    continue
                pts = positions[role_nodes]
                centroid = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - centroid, axis=1))) if len(pts) > 1 else 0.5
                lod_clusters.append({
                    'id': cluster_id,
                    'dataset': s_name,
                    'role': role_name,
                    'node_count': len(role_nodes),
                    'x': round(float(centroid[0]), 4),
                    'y': round(float(centroid[1]), 4),
                    'z': round(float(centroid[2]), 4),
                    'radius': round(radius, 4),
                    'sample_node_ids': role_nodes[:10],
                })
                cluster_id += 1

        # Generate 3D Causal & Information Flow Streamlines between neuropil clusters
        streamlines = []
        if len(lod_clusters) >= 2:
            sensory_c = [c for c in lod_clusters if c['role'] == 'sensory']
            central_c = [c for c in lod_clusters if c['role'] == 'central']
            motor_c = [c for c in lod_clusters if c['role'] == 'motor']
            for s in sensory_c[:4]:
                for c in central_c[:3]:
                    streamlines.append({
                        'source_cluster': s['id'],
                        'target_cluster': c['id'],
                        'flow_type': 'sensory_to_central',
                        'p0': [s['x'], s['y'], s['z']],
                        'p1': [c['x'], c['y'], c['z']],
                        'gain': 0.88,
                    })
            for c in central_c[:3]:
                for m in motor_c[:3]:
                    streamlines.append({
                        'source_cluster': c['id'],
                        'target_cluster': m['id'],
                        'flow_type': 'central_to_motor',
                        'p0': [c['x'], c['y'], c['z']],
                        'p1': [m['x'], m['y'], m['z']],
                        'gain': 0.94,
                    })

        return {
            'total_nodes': n,
            'total_edges': base_count + active_graft_count,
            'displayed_edges': len(edges),
            'base_edges': base_count,
            'active_graft_edges': active_graft_count,
            'graft_links': graft_links,
            'lod_clusters': lod_clusters,
            'num_lod_clusters': len(lod_clusters),
            'streamlines': streamlines,
            'num_streamlines': len(streamlines),
            'layout_scope': 'schematic source/role layout; coordinates are not measured anatomy',
            'nodes': nodes,
            'edges': edges,
            'neuropils': neuropils_list,
        }

    def get_live_activity_payload(self) -> Dict[str, Any]:
        """Returns real-time node firing rates, astrocyte dynamics, and neuromodulators."""
        r = self.agent.brain.r.detach().cpu().numpy()
        fe = self.agent.predictive.state.free_energy if hasattr(self.agent, 'predictive') else 0.0
        nm = self.agent.neuromod.state.to_dict() if hasattr(self.agent, 'neuromod') else {}
        astro = self.agent.astrocytes.summary() if hasattr(self.agent, 'astrocytes') else {}

        return {
            'ticks': getattr(self.agent, 'ticks', 0),
            'mean_rate': round(float(np.mean(r)), 4),
            'max_rate': round(float(np.max(r)), 4),
            'node_activations': np.round(r.astype(np.float64), 4).tolist(),
            'free_energy': round(float(fe), 4),
            'neuromodulators': nm,
            'astrocytes': astro,
        }
