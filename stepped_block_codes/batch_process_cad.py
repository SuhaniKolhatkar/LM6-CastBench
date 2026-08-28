import FreeCAD as App
import Part
import Sketcher
import Import   # headless-safe STEP exporter (works in FreeCADCmd, no GUI needed)
import csv
import os

# ---------------------------------------------------------------------
# CONFIG - edit these two paths
# ---------------------------------------------------------------------
CSV_PATH   = r"D:\NeurIPS\surrogate_block\doe.csv"
OUTPUT_DIR = r"D:\NeurIPS\surrogate_block\block_CAD_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_layer(doc, body, sketch_name, pad_name, l, b, t, z_offset):
    """Create one rectangular sketch on the XY plane (offset in Z) and pad it by t."""
    sketch = body.newObject('Sketcher::SketchObject', sketch_name)
    sketch.AttachmentSupport = [(doc.XY_Plane, '')]
    sketch.MapMode = 'FlatFace'
    sketch.AttachmentOffset = App.Placement(App.Vector(0, 0, z_offset), App.Rotation(0, 0, 0, 1))

    geoList = [
        Part.LineSegment(App.Vector(0, 0, 0), App.Vector(l, 0, 0)),
        Part.LineSegment(App.Vector(l, 0, 0), App.Vector(l, b, 0)),
        Part.LineSegment(App.Vector(l, b, 0), App.Vector(0, b, 0)),
        Part.LineSegment(App.Vector(0, b, 0), App.Vector(0, 0, 0)),
    ]
    sketch.addGeometry(geoList, False)

    constraints = [
        Sketcher.Constraint('Coincident', 0, 2, 1, 1),
        Sketcher.Constraint('Coincident', 1, 2, 2, 1),
        Sketcher.Constraint('Coincident', 2, 2, 3, 1),
        Sketcher.Constraint('Coincident', 3, 2, 0, 1),
        Sketcher.Constraint('Horizontal', 0),
        Sketcher.Constraint('Horizontal', 2),
        Sketcher.Constraint('Vertical', 1),
        Sketcher.Constraint('Vertical', 3),
    ]
    sketch.addConstraint(constraints)
    sketch.addConstraint(Sketcher.Constraint('Distance', 1, 1, 3, 2, l))
    sketch.addConstraint(Sketcher.Constraint('Distance', 0, 1, 2, 2, b))
    sketch.addConstraint(Sketcher.Constraint('Coincident', 0, 1, -1, 1))

    doc.recompute()

    pad = body.newObject('PartDesign::Pad', pad_name)
    pad.Profile = sketch
    pad.Length = t
    pad.Direction = (0, 0, 1)
    pad.ReferenceAxis = (sketch, ['N_Axis'])
    pad.Type = 0
    pad.Reversed = 0
    pad.Midplane = 0
    pad.Offset = 0

    doc.recompute()
    sketch.Visibility = False

    return pad


def build_stepped_block(row_index, l1, b1, t1, l2, b2, t2, l3, b3, t3):
    doc_name = f"stepped_block_{row_index:03d}"
    doc = App.newDocument(doc_name)
    body = doc.addObject('PartDesign::Body', 'Body')

    pad1 = make_layer(doc, body, 'Sketch',    'Pad',    l1, b1, t1, 0)
    pad2 = make_layer(doc, body, 'Sketch001', 'Pad001', l2, b2, t2, t1)
    pad3 = make_layer(doc, body, 'Sketch002', 'Pad002', l3, b3, t3, t1 + t2)

    pad1.Visibility = False
    pad2.Visibility = False

    doc.recompute()

    fcstd_path = os.path.join(OUTPUT_DIR, doc_name + ".FCStd")
    step_path  = os.path.join(OUTPUT_DIR, doc_name + ".step")

    doc.saveAs(fcstd_path)
    Import.export([body], step_path)

    App.closeDocument(doc.Name)
    print(f"Row {row_index}: saved {doc_name}.FCStd and {doc_name}.step")


def main():
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            l1, b1, t1 = float(row['l1']), float(row['b1']), float(row['t1'])
            l2, b2, t2 = float(row['l2']), float(row['b2']), float(row['t2'])
            l3, b3, t3 = float(row['l3']), float(row['b3']), float(row['t3'])
            build_stepped_block(i, l1, b1, t1, l2, b2, t2, l3, b3, t3)


if __name__ == "__main__":
    main()