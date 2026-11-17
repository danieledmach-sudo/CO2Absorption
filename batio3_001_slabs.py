#!/usr/bin/env python3
"""
batio3_001_slabs.py
Build (001) slabs of tetragonal P4mm BaTiO3 (or any ABO3 perovskite) from a CIF,
with a chosen termination on top: TiO2 or BaO.

Dependencies: Python 3.8+ and NumPy only (CIF is parsed internally, so it works
with Materials Project CIFs in either the "symmetrized" or the "conventional/P1"
form - symmetry operations in the file are applied before slab building).

Usage examples
--------------
  # Both terminations, 10 x 10 in-plane, 10 unit cells thick, 15 A vacuum
  python3 batio3_001_slabs.py BaTiO3.cif

  # Only the TiO2-terminated slab, 12 cells thick, 20 A vacuum
  python3 batio3_001_slabs.py BaTiO3.cif --top TiO2 --nz 12 --vacuum 20

  # Symmetric slab (same termination on top AND bottom; non-stoichiometric)
  python3 batio3_001_slabs.py BaTiO3.cif --symmetric

  # Fix the bottom 4 atomic layers (selective dynamics in the POSCAR)
  python3 batio3_001_slabs.py BaTiO3.cif --fix-bottom 4

Outputs (per termination):  <prefix>_001_<TOP>-top.cif  and  .vasp (POSCAR)

Notes
-----
* (001) = planes perpendicular to c, which in the standard P4mm setting is the
  4-fold / polar axis. The polarization therefore points along the surface
  normal. Use --flip-polarization to build the slab with P pointing the other
  way (into the slab instead of out of it).
* The default slab is stoichiometric (nz BaO + nz TiO2 layers): the chosen
  termination is on top, the other one is on the bottom. Such slabs carry a
  net dipole - use a dipole correction in DFT (e.g. VASP LDIPOL/IDIPOL=3).
"""
import argparse
import re
import sys
from collections import Counter, OrderedDict

import numpy as np

# ----------------------------------------------------------------------------
# CIF parsing
# ----------------------------------------------------------------------------
TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def _num(s):
    """'4.0123(4)' -> 4.0123"""
    return float(re.sub(r"\(\d+\)", "", s))


def _unquote(s):
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def read_cif(path):
    """Return (scalar tags dict, list of loops [ (headers, rows) ])."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read().splitlines()

    lines = []
    in_text = False
    for line in raw:
        if line.startswith(";"):          # multi-line text field - skip it
            in_text = not in_text
            continue
        if in_text:
            continue
        # strip comments that are not inside quotes
        toks = TOKEN.findall(line)
        clean = []
        for t in toks:
            if t.startswith("#"):
                break
            clean.append(t)
        if clean:
            lines.append(clean)

    tags, loops = {}, []
    i = 0
    while i < len(lines):
        toks = lines[i]
        if toks[0].lower() == "loop_":
            headers, values = [], []
            i += 1
            while i < len(lines) and lines[i][0].startswith("_"):
                headers.append(lines[i][0].lower())
                values.extend(lines[i][1:])   # rare: value on header line
                i += 1
            while (i < len(lines) and not lines[i][0].startswith("_")
                   and lines[i][0].lower() != "loop_"
                   and not lines[i][0].lower().startswith("data_")):
                values.extend(lines[i])
                i += 1
            n = len(headers)
            rows = [[_unquote(v) for v in values[k:k + n]]
                    for k in range(0, len(values) - n + 1, n)]
            loops.append((headers, rows))
            continue
        if toks[0].startswith("_"):
            if len(toks) > 1:
                tags[toks[0].lower()] = _unquote(" ".join(toks[1:]))
            elif i + 1 < len(lines) and not lines[i + 1][0].startswith("_"):
                tags[toks[0].lower()] = _unquote(" ".join(lines[i + 1]))
                i += 1
        i += 1
    return tags, loops


def find_loop(loops, *candidates):
    for headers, rows in loops:
        for c in candidates:
            if c in headers:
                return headers, rows
    return None, None


def parse_symop(op):
    """'x+1/2, -y, z' -> (3x3 rotation, translation)"""
    op = op.replace(" ", "").lower()
    parts = op.split(",")
    if len(parts) != 3:
        raise ValueError("Cannot parse symmetry operation: " + op)
    R, t = np.zeros((3, 3)), np.zeros(3)
    for row, expr in enumerate(parts):
        for sign, body in re.findall(r"([+-]?)([^+-]+)", expr):
            s = -1.0 if sign == "-" else 1.0
            m = re.fullmatch(r"(\d*\.?\d*(?:/\d+)?)\*?([xyz])", body)
            if m:
                coef = m.group(1)
                if coef in ("", None):
                    val = 1.0
                elif "/" in coef:
                    a, b = coef.split("/")
                    val = float(a) / float(b)
                else:
                    val = float(coef)
                R[row, "xyz".index(m.group(2))] += s * val
            else:
                if "/" in body:
                    a, b = body.split("/")
                    t[row] += s * float(a) / float(b)
                else:
                    t[row] += s * float(body)
    return R, t


def element_from(label):
    m = re.match(r"([A-Z][a-z]?)", label.strip())
    if not m:
        raise ValueError("Cannot get element from site label: " + label)
    return m.group(1)


def load_structure(path):
    tags, loops = read_cif(path)
    try:
        a, b, c = (_num(tags["_cell_length_" + k]) for k in "abc")
        al, be, ga = (_num(tags["_cell_angle_" + k]) for k in ("alpha", "beta", "gamma"))
    except KeyError as exc:
        sys.exit("ERROR: missing cell parameter in CIF: {}".format(exc))

    sg = (tags.get("_symmetry_space_group_name_h-m")
          or tags.get("_space_group_name_h-m_alt") or "?")
    sg_num = (tags.get("_symmetry_int_tables_number")
              or tags.get("_space_group_it_number") or "?")

    # symmetry operations
    h, rows = find_loop(loops, "_symmetry_equiv_pos_as_xyz",
                        "_space_group_symop_operation_xyz")
    ops = []
    if h:
        col = (h.index("_symmetry_equiv_pos_as_xyz")
               if "_symmetry_equiv_pos_as_xyz" in h
               else h.index("_space_group_symop_operation_xyz"))
        ops = [parse_symop(r[col]) for r in rows]
    if not ops:
        ops = [(np.eye(3), np.zeros(3))]

    # atom sites
    h, rows = find_loop(loops, "_atom_site_fract_x")
    if not h:
        sys.exit("ERROR: no _atom_site_fract_x loop found in CIF")
    ix, iy, iz = (h.index("_atom_site_fract_" + k) for k in "xyz")
    it = h.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in h else None
    il = h.index("_atom_site_label") if "_atom_site_label" in h else None
    io = h.index("_atom_site_occupancy") if "_atom_site_occupancy" in h else None

    species, frac = [], []
    for r in rows:
        el = element_from(r[it] if it is not None else r[il])
        if io is not None and r[io] not in ("?", ".") and abs(_num(r[io]) - 1) > 1e-3:
            sys.exit("ERROR: partial occupancy on site {} - not supported".format(r[il]))
        p = np.array([_num(r[ix]), _num(r[iy]), _num(r[iz])])
        for R, t in ops:                       # apply all symmetry operations
            q = (R @ p + t) % 1.0
            dup = False
            for s2, f2 in zip(species, frac):
                d = q - f2
                d -= np.round(d)
                if s2 == el and np.all(np.abs(d) < 1e-3):
                    dup = True
                    break
            if not dup:
                species.append(el)
                frac.append(q)
    return dict(abc=(a, b, c), angles=(al, be, ga), sg=sg, sg_num=sg_num,
                species=species, frac=np.array(frac))


# ----------------------------------------------------------------------------
# Layer analysis
# ----------------------------------------------------------------------------
def parse_formula(f):
    out = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        out[el] += int(n) if n else 1
    if not out:
        raise ValueError("Bad formula: " + f)
    return out


def reduced(counter):
    g = 0
    for v in counter.values():
        g = np.gcd(g, v)
    return Counter({k: v // g for k, v in counter.items()})


def find_layers(species, frac_z, c_len, tol):
    """Group atoms into (001) atomic layers using a periodic gap criterion.
    Returns list of dicts: {idx: [...], center: z_frac, comp: Counter}."""
    order = np.argsort(frac_z)
    z = frac_z[order]
    n = len(z)
    gaps = np.diff(np.concatenate([z, [z[0] + 1.0]]))     # gap after atom k
    start = (int(np.argmax(gaps)) + 1) % n                # begin after largest gap
    layers, cur, cur_z = [], [order[start]], [z[start]]
    for step in range(1, n):
        k = (start + step) % n
        prev = (start + step - 1) % n
        if gaps[prev] * c_len > tol:
            layers.append((cur, cur_z))
            cur, cur_z = [], []
        zz = z[k] + (1.0 if k < start else 0.0)           # unwrap across boundary
        cur.append(order[k])
        cur_z.append(zz)
    layers.append((cur, cur_z))
    out = []
    for idx, zs in layers:
        out.append(dict(idx=list(idx), center=float(np.mean(zs)) % 1.0,
                        span=(max(zs) - min(zs)) * c_len,
                        comp=Counter(species[i] for i in idx)))
    return out


# ----------------------------------------------------------------------------
# Slab construction
# ----------------------------------------------------------------------------
def build_slab(st, top_formula, nx, ny, nz, vacuum, symmetric, flip, tol):
    a, b, c = st["abc"]
    al, be, ga = st["angles"]
    if abs(al - 90) > 0.5 or abs(be - 90) > 0.5:
        sys.exit("ERROR: alpha/beta are not 90 deg; c is not normal to (001). "
                 "Expected a tetragonal P4mm cell.")
    species = list(st["species"])
    frac = st["frac"].copy()

    if flip:                     # reverse the polar axis: z -> -z
        frac[:, 2] = (-frac[:, 2]) % 1.0

    layers = find_layers(species, frac[:, 2], c, tol)
    if len(layers) != 2:
        desc = ", ".join("{}@{:.3f}".format(dict(L["comp"]), L["center"]) for L in layers)
        sys.exit("ERROR: expected 2 (001) layers per unit cell, found {}: {}\n"
                 "Try adjusting --layer-tol (current {} A).".format(len(layers), desc, tol))

    top_red = reduced(parse_formula(top_formula))
    match = [i for i, L in enumerate(layers) if reduced(L["comp"]) == top_red]
    if len(match) != 1:
        sys.exit("ERROR: no unique layer with composition {} (layers: {})".format(
            top_formula, [dict(L["comp"]) for L in layers]))
    top_L = layers[match[0]]
    bot_L = layers[1 - match[0]]

    # Shift so the bottom-type layer sits near z=0.25 and the top-type near 0.75
    shift = 0.25 - bot_L["center"]
    fz = (frac[:, 2] + shift) % 1.0
    if not (all(0 < fz[i] < 0.5 for i in bot_L["idx"]) and
            all(0.5 < fz[i] < 1.0 for i in top_L["idx"])):
        sys.exit("ERROR: layers too thick to separate cleanly within one cell.")

    # Name of the bottom-type layer, cations first (e.g. BaO / TiO2)
    red = reduced(bot_L["comp"])
    bot_name = "".join(e + (str(red[e]) if red[e] > 1 else "")
                       for e in sorted(red, key=lambda e: (e == "O", e)))

    # Stack: k = 0..nz-1 gives bottom,top,bottom,top... (top termination last)
    blocks = []                                  # (element, fx, fy, z_cart, layer_id)
    seq = []
    k_layers = [(-1, top_L)] if symmetric else []
    for k in range(nz):
        k_layers += [(k, bot_L), (k, top_L)]
    for layer_id, (k, L) in enumerate(k_layers):
        seq.append(top_formula if L is top_L else bot_name)
        for i in L["idx"]:
            for ix in range(nx):
                for iy in range(ny):
                    blocks.append((species[i],
                                   (frac[i, 0] + ix) / nx,
                                   (frac[i, 1] + iy) / ny,
                                   (fz[i] + k) * c,
                                   layer_id))

    zc = np.array([x[3] for x in blocks])
    zmin, zmax = zc.min(), zc.max()
    thick = zmax - zmin
    c_new = thick + vacuum

    # In-plane lattice (gamma may be 90 for P4mm, but keep it general)
    g = np.radians(ga)
    A = np.array([a * nx, 0, 0])
    B = np.array([b * ny * np.cos(g), b * ny * np.sin(g), 0])
    C = np.array([0, 0, c_new])

    atoms = []
    for el, fx, fy, z, lid in blocks:
        fzn = (z - zmin + vacuum / 2.0) / c_new   # slab centred in the cell
        atoms.append((el, fx, fy, fzn, lid))
    # order by element (first appearance), then z
    elem_order = list(OrderedDict.fromkeys(species))      # same order as input CIF
    atoms.sort(key=lambda t: (elem_order.index(t[0]), t[3], t[1], t[2]))
    return dict(lattice=np.array([A, B, C]), atoms=atoms, elem_order=elem_order,
                thickness=thick, c_new=c_new, sequence=seq, n_layers=len(k_layers))


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
def write_cif(path, slab, title):
    L = slab["lattice"]
    a, b, c = (np.linalg.norm(v) for v in L)
    ang = lambda u, v: np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    al, be, ga = ang(L[1], L[2]), ang(L[0], L[2]), ang(L[0], L[1])
    vol = abs(np.linalg.det(L))
    comp = Counter(x[0] for x in slab["atoms"])
    with open(path, "w") as f:
        f.write("# {}\n# generated by batio3_001_slabs.py\n".format(title))
        f.write("data_{}\n".format(re.sub(r"\W", "_", title)))
        f.write("_symmetry_space_group_name_H-M   'P 1'\n_symmetry_Int_Tables_number   1\n")
        for k, v in zip(("a", "b", "c"), (a, b, c)):
            f.write("_cell_length_{}   {:.6f}\n".format(k, v))
        for k, v in zip(("alpha", "beta", "gamma"), (al, be, ga)):
            f.write("_cell_angle_{}   {:.6f}\n".format(k, v))
        f.write("_cell_volume   {:.4f}\n".format(vol))
        f.write("_chemical_formula_sum   '{}'\n".format(
            " ".join("{}{}".format(e, comp[e]) for e in slab["elem_order"])))
        f.write("loop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n")
        f.write("loop_\n _atom_site_label\n _atom_site_type_symbol\n"
                " _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n"
                " _atom_site_occupancy\n")
        cnt = Counter()
        for el, x, y, z, _ in slab["atoms"]:
            cnt[el] += 1
            f.write("  {}{}  {}  {:.6f}  {:.6f}  {:.6f}  1.0\n".format(el, cnt[el], el, x, y, z))


def write_poscar(path, slab, title, fix_bottom):
    L = slab["lattice"]
    comp = Counter(x[0] for x in slab["atoms"])
    with open(path, "w") as f:
        f.write(title + "\n1.0\n")
        for v in L:
            f.write("  {:.10f}  {:.10f}  {:.10f}\n".format(*v))
        f.write("  " + "  ".join(slab["elem_order"]) + "\n")
        f.write("  " + "  ".join(str(comp[e]) for e in slab["elem_order"]) + "\n")
        if fix_bottom > 0:
            f.write("Selective dynamics\n")
        f.write("Direct\n")
        for el, x, y, z, lid in slab["atoms"]:
            line = "  {:.10f}  {:.10f}  {:.10f}".format(x, y, z)
            if fix_bottom > 0:
                line += "   F F F" if lid < fix_bottom else "   T T T"
            f.write(line + "   " + el + "\n")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Build TiO2- and/or BaO-terminated (001) slabs of P4mm BaTiO3 from a CIF.")
    p.add_argument("cif", help="input CIF (e.g. from Materials Project)")
    p.add_argument("--top", choices=["TiO2", "BaO", "both"], default="both",
                   help="termination on top of the slab (default: both)")
    p.add_argument("--nx", type=int, default=10, help="repeats along a (default 10)")
    p.add_argument("--ny", type=int, default=10, help="repeats along b (default 10)")
    p.add_argument("--nz", type=int, default=10,
                   help="unit cells (BaO+TiO2 bilayers) along c (default 10)")
    p.add_argument("--vacuum", type=float, default=15.0, help="vacuum thickness in A (default 15)")
    p.add_argument("--symmetric", action="store_true",
                   help="add one extra termination layer at the bottom so both "
                        "surfaces have the same termination (non-stoichiometric)")
    p.add_argument("--flip-polarization", action="store_true",
                   help="invert the polar c axis before building the slab")
    p.add_argument("--fix-bottom", type=int, default=0,
                   help="fix this many bottom atomic layers in the POSCAR (default 0)")
    p.add_argument("--layer-tol", type=float, default=0.6,
                   help="max z gap (A) between atoms of the same layer (default 0.6)")
    p.add_argument("--prefix", default="BaTiO3", help="output file prefix")
    args = p.parse_args()

    if min(args.nx, args.ny, args.nz) < 10:
        print("WARNING: requested size is smaller than 10x10x10 unit cells.")

    st = load_structure(args.cif)
    sg_clean = st["sg"].replace(" ", "").upper()
    print("Read {}: {} atoms/cell, a={:.4f} b={:.4f} c={:.4f} A, space group {} (#{})".format(
        args.cif, len(st["species"]), *st["abc"], st["sg"], st["sg_num"]))
    if sg_clean != "P4MM" and str(st["sg_num"]) != "99":
        print("WARNING: space group is not P4mm (#99); check that c is the stacking axis.")
    comp = Counter(st["species"])
    print("Unit-cell composition:", dict(comp))
    a, b, c = st["abc"]
    if abs(a - b) > 1e-3:
        print("WARNING: a != b; this does not look like a standard tetragonal cell.")

    tops = ["TiO2", "BaO"] if args.top == "both" else [args.top]
    for top in tops:
        slab = build_slab(st, top, args.nx, args.ny, args.nz, args.vacuum,
                          args.symmetric, args.flip_polarization, args.layer_tol)
        tag = "{}_001_{}-top{}".format(args.prefix, top, "_sym" if args.symmetric else "")
        title = "{} (001) {} {}x{}x{} slab{}".format(
            args.prefix, top + "-terminated", args.nx, args.ny, args.nz,
            " symmetric" if args.symmetric else "")
        write_cif(tag + ".cif", slab, title)
        write_poscar(tag + ".vasp", slab, title, args.fix_bottom)
        comp = Counter(x[0] for x in slab["atoms"])
        print("\n== {} ==".format(title))
        print("  atoms: {}  ({})".format(len(slab["atoms"]),
              " ".join("{}{}".format(e, comp[e]) for e in slab["elem_order"])))
        print("  layers (bottom -> top): {} ... {}  [{} layers]".format(
            "/".join(slab["sequence"][:3]), "/".join(slab["sequence"][-3:]), slab["n_layers"]))
        print("  in-plane: {:.3f} x {:.3f} A,  slab thickness {:.3f} A,  c = {:.3f} A".format(
            np.linalg.norm(slab["lattice"][0]), np.linalg.norm(slab["lattice"][1]),
            slab["thickness"], slab["c_new"]))
        print("  wrote {}.cif and {}.vasp".format(tag, tag))
    if not args.symmetric:
        print("\nNote: stoichiometric asymmetric slabs carry a dipole along z; "
              "use a dipole correction in periodic calculations.")


if __name__ == "__main__":
    main()
