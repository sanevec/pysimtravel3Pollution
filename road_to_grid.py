#!/usr/bin/env python3
"""
road_to_grid.py
---------------
Convierte una red viaria (formato .gpkg o .graphml de OSMnx) en una matriz
de celdas con información de carretera, velocidad y conectividad.

Uso:
    python road_to_grid.py --input graph.gpkg --cell-size 25
    python road_to_grid.py --input graph.graphml --cell-size 50 --output mi_grid
    python road_to_grid.py --input graph.gpkg --cell-size 25 --formats json csv visual

Dependencias:
    pip install geopandas fiona networkx shapely numpy
"""

import argparse
import json
import csv
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Dependencias opcionales con mensajes claros ────────────────────────────────

try:
    import geopandas as gpd
except ImportError:
    sys.exit("ERROR: Instala geopandas:  pip install geopandas fiona")

try:
    import fiona
except ImportError:
    sys.exit("ERROR: Instala fiona:  pip install fiona")


# ── Carga de archivos ──────────────────────────────────────────────────────────

def load_gpkg(path: Path):
    """Carga nodos y aristas desde un .gpkg generado por OSMnx."""
    layers = fiona.listlayers(str(path))
    if "edges" not in layers:
        sys.exit(f"ERROR: El archivo {path} no contiene una capa 'edges'.")
    nodes = gpd.read_file(str(path), layer="nodes") if "nodes" in layers else None
    edges = gpd.read_file(str(path), layer="edges")
    return nodes, edges


def load_graphml(path: Path):
    """Carga nodos y aristas desde un .graphml generado por OSMnx."""
    try:
        import networkx as nx
    except ImportError:
        sys.exit("ERROR: Instala networkx:  pip install networkx")

    G = nx.read_graphml(str(path))

    # Nodos
    node_records = []
    for node_id, data in G.nodes(data=True):
        node_records.append({
            "osmid": node_id,
            "x": float(data.get("x", 0)),
            "y": float(data.get("y", 0)),
        })

    # Aristas
    edge_records = []
    for u, v, data in G.edges(data=True):
        # speed_kph puede venir como string o float
        speed_raw = data.get("speed_kph", 30)
        try:
            speed = float(speed_raw)
        except (ValueError, TypeError):
            speed = 30.0

        oneway_raw = data.get("oneway", "False")
        oneway = oneway_raw in (True, "True", "true", "1")

        reversed_raw = data.get("reversed", "False")
        reversed_ = reversed_raw in (True, "True", "true", "1")

        edge_records.append({
            "u": u, "v": v,
            "highway": data.get("highway", ""),
            "speed_kph": speed,
            "oneway": oneway,
            "reversed": reversed_,
            "length": float(data.get("length", 0)),
            "geometry": None,  # se reconstruye desde nodos
        })

    # Reconstruir geometrías lineales desde coordenadas de nodos
    from shapely.geometry import LineString, Point
    node_xy = {r["osmid"]: (r["x"], r["y"]) for r in node_records}

    import pandas as pd
    edge_gdf_records = []
    for e in edge_records:
        u_xy = node_xy.get(str(e["u"]))
        v_xy = node_xy.get(str(e["v"]))
        if u_xy and v_xy:
            geom = LineString([u_xy, v_xy])
        else:
            geom = None
        edge_gdf_records.append({**e, "geometry": geom})

    edges = gpd.GeoDataFrame(edge_gdf_records, geometry="geometry", crs="EPSG:4326")
    nodes = gpd.GeoDataFrame(node_records, geometry=gpd.points_from_xy(
        [r["x"] for r in node_records],
        [r["y"] for r in node_records]
    ), crs="EPSG:4326")

    return nodes, edges


def load_network(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        return load_gpkg(path)
    elif suffix in (".graphml", ".xml"):
        return load_graphml(path)
    else:
        sys.exit(f"ERROR: Formato no soportado '{suffix}'. Usa .gpkg o .graphml")


# ── Construcción de la malla ───────────────────────────────────────────────────

def point_to_cell(x, y, minx, miny, cell_size):
    c = int((x - minx) / cell_size)
    r = int((y - miny) / cell_size)
    return (r, c)


def cells_along_linestring(geom, minx, miny, cell_size, step_m=5):
    """Muestrea puntos a lo largo de una LineString y devuelve celdas ordenadas."""
    length = geom.length
    if length == 0:
        pt = geom.interpolate(0)
        return [point_to_cell(pt.x, pt.y, minx, miny, cell_size)]

    distances = np.arange(0, length, step_m)
    cells = []
    prev = None
    for d in distances:
        pt = geom.interpolate(d)
        cell = point_to_cell(pt.x, pt.y, minx, miny, cell_size)
        if cell != prev:
            cells.append(cell)
            prev = cell

    # Asegurar que el punto final está incluido
    pt = geom.interpolate(length)
    cell = point_to_cell(pt.x, pt.y, minx, miny, cell_size)
    if not cells or cell != cells[-1]:
        cells.append(cell)

    return cells

import pandas as pd

def get_num_lanes(edge):
    lanes_attr = edge.get("lanes")
    if pd.notna(lanes_attr):
        if isinstance(lanes_attr, list):
            lanes_attr = lanes_attr[0]
        try:
            return int(lanes_attr)
        except:
            pass
    highway = str(edge.get("highway", ""))
    if "motorway" in highway or "trunk" in highway:
        return 3
    elif "primary" in highway or "secondary" in highway:
        return 2
    else:
        return 1

def get_lane_geometries(geom, oneway, lanes, cell_size):
    # Devuelve [(geom, is_forward), ...]
    lane_width = cell_size
    results = []
    
    if oneway:
        if lanes == 1:
            offsets = [0.0]
        else:
            start = -(lanes - 1) * lane_width / 2.0
            offsets = [start + i * lane_width for i in range(lanes)]
            
        for off in offsets:
            try:
                g = geom.offset_curve(off) if off != 0 else geom
                if g.geom_type == 'LineString' and not g.is_empty:
                    results.append((g, True))
                elif g.geom_type == 'MultiLineString':
                    for part in g.geoms:
                        results.append((part, True))
            except:
                results.append((geom, True))
        if not results:
            results = [(geom, True)]
    else:
        lanes_each_way = max(1, lanes // 2)
        fwd_offsets = [-lane_width / 2.0 - i * lane_width for i in range(lanes_each_way)]
        bwd_offsets = [lane_width / 2.0 + i * lane_width for i in range(lanes_each_way)]
        
        for off in fwd_offsets:
            try:
                g = geom.offset_curve(off)
                if g.geom_type == 'LineString' and not g.is_empty:
                    results.append((g, True))
                elif g.geom_type == 'MultiLineString':
                    for part in g.geoms:
                        results.append((part, True))
            except:
                results.append((geom, True))
                
        for off in bwd_offsets:
            try:
                g = geom.offset_curve(off)
                if g.geom_type == 'LineString' and not g.is_empty:
                    results.append((g, False))
                elif g.geom_type == 'MultiLineString':
                    for part in g.geoms:
                        results.append((part, False))
            except:
                results.append((geom, False))
                
        if not results:
            results = [(geom, True), (geom, False)]
            
    return results


def build_grid(edges: gpd.GeoDataFrame, cell_size: float):
    """
    Construye la matriz de celdas a partir de las aristas.

    Parámetros
    ----------
    edges      : GeoDataFrame con columnas geometry, speed_kph, oneway, reversed
    cell_size  : tamaño de celda en metros

    Devuelve
    --------
    grid_info  : dict con metadatos de la malla
    cell_road  : dict (r,c) -> True
    cell_speed : dict (r,c) -> float (kph)
    cell_highway : dict (r,c) -> str
    cell_neighbors : dict (r,c) -> set of (r,c)
    rows, cols : dimensiones de la malla
    minx, miny : origen en metros (EPSG:32630)
    """

    # Proyectar a métrico para que las distancias sean reales
    edges_m = edges.to_crs("EPSG:32630")

    minx, miny, maxx, maxy = edges_m.total_bounds
    # Margen de una celda en cada lado
    minx -= cell_size; miny -= cell_size
    maxx += cell_size; maxy += cell_size

    cols = int(np.ceil((maxx - minx) / cell_size))
    rows = int(np.ceil((maxy - miny) / cell_size))

    cell_road      = {}
    cell_speed     = {}
    cell_highway   = {}
    cell_neighbors = defaultdict(set)

    for _, edge in edges_m.iterrows():
        geom = edge.geometry
        if geom is None or geom.is_empty:
            continue

        speed   = float(edge.get("speed_kph", 30) or 30)
        oneway  = bool(edge.get("oneway", False))
        rev     = bool(edge.get("reversed", False))
        highway = str(edge.get("highway", ""))
        lanes   = get_num_lanes(edge)

        # Nodos originales para intersecciones perfectas
        start_coord = geom.coords[0]
        end_coord = geom.coords[-1]
        start_cell = point_to_cell(start_coord[0], start_coord[1], minx, miny, cell_size)
        end_cell = point_to_cell(end_coord[0], end_coord[1], minx, miny, cell_size)

        lane_geoms = get_lane_geometries(geom, oneway, lanes, cell_size)

        # Interpolación de Bresenham para celdas de cuadrícula
        def bresenham(r0, c0, r1, c1):
            cells = []
            dc = abs(c1 - c0)
            dr = abs(r1 - r0)
            sc = 1 if c0 < c1 else -1
            sr = 1 if r0 < r1 else -1
            err = dc - dr
            while True:
                cells.append((r0, c0))
                if r0 == r1 and c0 == c1:
                    break
                e2 = 2 * err
                if e2 > -dr:
                    err -= dr
                    c0 += sc
                if e2 < dc:
                    err += dc
                    r0 += sr
            return cells

        for lane_geom, is_forward in lane_geoms:
            raw_seq = cells_along_linestring(lane_geom, minx, miny, cell_size)

            # Rellenar cualquier hueco (salto > 1 celda) asegurando que empiece y acabe en la intersección exacta
            full_raw_seq = [start_cell] + raw_seq + [end_cell]
            seq = []
            for i in range(len(full_raw_seq) - 1):
                c1 = full_raw_seq[i]
                c2 = full_raw_seq[i+1]
                path = bresenham(c1[0], c1[1], c2[0], c2[1])
                if not seq:
                    seq.extend(path)
                else:
                    if path[0] == seq[-1]:
                        seq.extend(path[1:])
                    else:
                        seq.extend(path)
            
            # Marcar celdas como carretera
            for cell in seq:
                cell_road[cell] = True
                if cell not in cell_speed or cell_speed[cell] < speed:
                    cell_speed[cell]   = speed
                    cell_highway[cell] = highway

            # Conectividad continua (start_cell -> seq -> end_cell ya integrados)
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if is_forward:
                    if rev:
                        cell_neighbors[b].add(a)
                    else:
                        cell_neighbors[a].add(b)
                else:
                    if rev:
                        cell_neighbors[a].add(b)
                    else:
                        cell_neighbors[b].add(a)

    grid_info = {
        "cell_size_m": cell_size,
        "rows": rows,
        "cols": cols,
        "origin_x_m": minx,
        "origin_y_m": miny,
        "crs": "EPSG:32630",
        "description": (
            "row 0 = borde sur, col 0 = borde oeste. "
            "neighbors lista las celdas accesibles en el sentido de circulación."
        )
    }

    return grid_info, cell_road, cell_speed, cell_highway, cell_neighbors, rows, cols


# ── Exportación ────────────────────────────────────────────────────────────────

def export_json(grid_info, cell_road, cell_speed, cell_highway, cell_neighbors,
                rows, cols, out_path: Path):
    """Exporta solo las celdas con carretera en JSON."""
    cells = []
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell_road.get(cell):
                cells.append({
                    "row": r,
                    "col": c,
                    "is_road": True,
                    "speed_kph": cell_speed.get(cell),
                    "highway": cell_highway.get(cell, ""),
                    "neighbors": sorted(list(cell_neighbors.get(cell, set()))),
                })

    data = {"grid_info": grid_info, "cells": cells}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON  → {out_path}  ({len(cells)} celdas de carretera)")


def export_csv(grid_info, cell_road, cell_speed, cell_highway, cell_neighbors,
               rows, cols, out_path: Path):
    """Exporta la malla completa en CSV (todas las celdas)."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "is_road", "speed_kph", "highway",
                         "neighbor_count", "neighbors"])
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                is_road = cell_road.get(cell, False)
                speed   = cell_speed.get(cell, "") if is_road else ""
                hw      = cell_highway.get(cell, "") if is_road else ""
                nbs     = sorted(list(cell_neighbors.get(cell, set()))) if is_road else []
                writer.writerow([r, c, is_road, speed, hw, len(nbs), nbs])
    total = rows * cols
    road  = sum(1 for v in cell_road.values() if v)
    print(f"  ✓ CSV   → {out_path}  ({total} celdas, {road} con carretera)")


def export_visual(grid_info, cell_road, cell_speed, rows, cols, out_path: Path):
    """Exporta un mapa ASCII con la velocidad de cada celda."""
    # Caracteres por rango de velocidad
    def speed_char(s):
        if s is None:   return "·"
        if s <= 30:     return "░"
        if s <= 40:     return "▒"
        if s <= 50:     return "▓"
        return "█"

    lines = [
        f"Malla {cols}x{rows} — celda = {grid_info['cell_size_m']}m",
        "· sin carretera  ░ ≤30kph  ▒ ≤40kph  ▓ ≤50kph  █ >50kph",
        "",
    ]
    for r in range(rows - 1, -1, -1):  # norte arriba
        line = "".join(speed_char(cell_speed.get((r, c))) for c in range(cols))
        lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Visual→ {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convierte una red viaria OSMnx (.gpkg/.graphml) en una matriz de celdas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Archivo de entrada (.gpkg o .graphml)"
    )
    parser.add_argument(
        "--cell-size", "-s", type=float, default=25,
        help="Tamaño de celda en metros (por defecto: 25)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Prefijo del archivo de salida (por defecto: nombre del input)"
    )
    parser.add_argument(
        "--formats", "-f", nargs="+",
        choices=["json", "csv", "visual"], default=["json", "csv", "visual"],
        help="Formatos de salida (por defecto: json csv visual)"
    )
    return parser.parse_args()


def prune_sinks(cell_road, cell_neighbors, cell_speed, cell_highway):
    print("\n🧹 Pruning sink nodes (dead ends)...")
    removed_total = 0
    iteration = 1
    while True:
        sinks = []
        for cell, is_road in cell_road.items():
            if is_road and len(cell_neighbors.get(cell, set())) == 0:
                sinks.append(cell)
        
        if not sinks:
            break
            
        print(f"   Iteration {iteration}: pruning {len(sinks)} sink nodes")
        for sink in sinks:
            cell_road[sink] = False
            if sink in cell_neighbors:
                del cell_neighbors[sink]
            if sink in cell_speed:
                del cell_speed[sink]
            if sink in cell_highway:
                del cell_highway[sink]
        
        for neighbors in cell_neighbors.values():
            neighbors.difference_update(sinks)
                
        removed_total += len(sinks)
        iteration += 1
        
    print(f"   Total pruned sink nodes: {removed_total}")
    return cell_road, cell_neighbors, cell_speed, cell_highway

import networkx as nx

def keep_largest_scc(cell_road, cell_neighbors, cell_speed, cell_highway):
    print("\n🧹 Keeping only the largest strongly connected component...")
    
    G = nx.DiGraph()
    for cell, is_road in cell_road.items():
        if is_road:
            G.add_node(cell)
            for neighbor in cell_neighbors.get(cell, set()):
                G.add_edge(cell, neighbor)
                
    sccs = list(nx.strongly_connected_components(G))
    sccs.sort(key=len, reverse=True)
    
    if not sccs:
        return cell_road, cell_neighbors, cell_speed, cell_highway
        
    largest_scc = sccs[0]
    removed_count = 0
    
    for cell, is_road in list(cell_road.items()):
        if is_road and cell not in largest_scc:
            cell_road[cell] = False
            if cell in cell_neighbors:
                del cell_neighbors[cell]
            if cell in cell_speed:
                del cell_speed[cell]
            if cell in cell_highway:
                del cell_highway[cell]
            removed_count += 1
            
    # Clean up neighbors
    for neighbors in cell_neighbors.values():
        neighbors.intersection_update(largest_scc)
        
    print(f"   Largest SCC size: {len(largest_scc)} cells")
    print(f"   Removed isolated cells: {removed_count}")
    
    return cell_road, cell_neighbors, cell_speed, cell_highway

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: No se encuentra el archivo '{input_path}'")

    out_prefix = Path(args.output) if args.output else input_path.with_suffix("")
    cell_size  = args.cell_size

    print(f"\n📂 Cargando {input_path.name} ...")
    nodes, edges = load_network(input_path)
    print(f"   {len(edges)} aristas cargadas")

    print(f"\n🔲 Construyendo malla de celdas ({cell_size}m × {cell_size}m) ...")
    grid_info, cell_road, cell_speed, cell_highway, cell_neighbors, rows, cols = \
        build_grid(edges, cell_size)

    cell_road, cell_neighbors, cell_speed, cell_highway = keep_largest_scc(cell_road, cell_neighbors, cell_speed, cell_highway)

    road_count = sum(1 for v in cell_road.values() if v)
    total = rows * cols
    print(f"   Malla: {cols} cols × {rows} rows = {total} celdas")
    print(f"   Celdas con carretera: {road_count} ({100*road_count/total:.1f}%)")
    speeds = list(cell_speed.values())
    if speeds:
        print(f"   Velocidades: {min(speeds):.0f} – {max(speeds):.0f} kph")

    print(f"\n💾 Exportando ...")
    if "json" in args.formats:
        export_json(grid_info, cell_road, cell_speed, cell_highway, cell_neighbors,
                    rows, cols, Path(str(out_prefix) + "_grid.json"))

    if "csv" in args.formats:
        export_csv(grid_info, cell_road, cell_speed, cell_highway, cell_neighbors,
                   rows, cols, Path(str(out_prefix) + "_grid.csv"))

    if "visual" in args.formats:
        export_visual(grid_info, cell_road, cell_speed, rows, cols,
                      Path(str(out_prefix) + "_grid_visual.txt"))

    print("\n✅ Listo.\n")


if __name__ == "__main__":
    main()
