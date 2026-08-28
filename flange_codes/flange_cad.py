# -*- coding: utf-8 -*-
"""
FreeCAD PartDesign macro - Flange with bore, tapered boss, and 4-hole bolt pattern
Cleaned/reconstructed from recorded macro.
"""

import FreeCAD as App
import Part
import Sketcher

# ---------------------------------------------------------------
# Document & Body
# ---------------------------------------------------------------
doc = App.newDocument("Flange")
body = doc.addObject('PartDesign::Body', 'Body')
doc.recompute()

# ---------------------------------------------------------------
# Sketch (base disc, dia 100) on XY_Plane -> Pad
# ---------------------------------------------------------------
sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
sketch.AttachmentSupport = (doc.getObject('XY_Plane'), [''])
sketch.MapMode = 'FlatFace'
doc.recompute()

sketch.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), 50.0), False)
sketch.addConstraint(Sketcher.Constraint('Diameter', 0, 100.0))
sketch.addConstraint(Sketcher.Constraint('Coincident', 0, 3, -1, 1))
doc.recompute()

pad = body.newObject('PartDesign::Pad', 'Pad')
pad.Profile = (sketch, [''])
pad.Length = 15.0
pad.TaperAngle = 0.0
pad.UseCustomVector = False
pad.Direction = (0, 0, 1)
pad.ReferenceAxis = (sketch, ['N_Axis'])
pad.AlongSketchNormal = True
pad.Type = 0
pad.Midplane = False
pad.Reversed = False
pad.Offset = 0.0
sketch.Visibility = False
doc.recompute()

# ---------------------------------------------------------------
# Sketch001 (dia 50) on Pad.Face3 -> Pocket
# ---------------------------------------------------------------
sketch001 = body.newObject('Sketcher::SketchObject', 'Sketch001')
sketch001.AttachmentSupport = (pad, ['Face3'])
sketch001.MapMode = 'FlatFace'
doc.recompute()

sketch001.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), 25.0), False)
sketch001.addConstraint(Sketcher.Constraint('Diameter', 0, 50.0))
sketch001.addConstraint(Sketcher.Constraint('Coincident', 0, 3, -1, 1))
doc.recompute()

pocket = body.newObject('PartDesign::Pocket', 'Pocket')
pocket.Profile = (sketch001, [''])
pocket.Length = 15.0
pocket.TaperAngle = 0.0
pocket.UseCustomVector = False
pocket.Direction = (0, 0, -1)
pocket.ReferenceAxis = (sketch001, ['N_Axis'])
pocket.AlongSketchNormal = True
pocket.Type = 0
pocket.Midplane = False
pocket.Reversed = False
pocket.Offset = 0.0
sketch001.Visibility = False
pad.Visibility = False
doc.recompute()

# ---------------------------------------------------------------
# Sketch002 (annulus, dia 50/60) on Pocket.Face3 -> Pad001 (tapered boss)
# ---------------------------------------------------------------
sketch002 = body.newObject('Sketcher::SketchObject', 'Sketch002')
sketch002.AttachmentSupport = (pocket, ['Face3'])
sketch002.MapMode = 'FlatFace'
doc.recompute()

sketch002.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), 25.0), False)
sketch002.addConstraint(Sketcher.Constraint('Diameter', 0, 50.0))
sketch002.addConstraint(Sketcher.Constraint('Coincident', 0, 3, -1, 1))

sketch002.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), 30.0), False)
sketch002.addConstraint(Sketcher.Constraint('Diameter', 1, 60.0))
sketch002.addConstraint(Sketcher.Constraint('Coincident', 1, 3, 0, 3))
doc.recompute()

pad001 = body.newObject('PartDesign::Pad', 'Pad001')
pad001.Profile = (sketch002, [''])
pad001.Length = 15.0
pad001.TaperAngle = -10.0
pad001.UseCustomVector = False
pad001.Direction = (0, 0, 1)
pad001.ReferenceAxis = (sketch002, ['N_Axis'])
pad001.AlongSketchNormal = True
pad001.Type = 0
pad001.Midplane = False
pad001.Reversed = False
pad001.Offset = 0.0
sketch002.Visibility = False
pocket.Visibility = False
doc.recompute()

# ---------------------------------------------------------------
# Sketch003 (4x bolt holes, dia 8, on 40mm radius bolt circle) on Pad001.Face3 -> Pocket001
# ---------------------------------------------------------------
sketch003 = body.newObject('Sketcher::SketchObject', 'Sketch003')
sketch003.AttachmentSupport = (pad001, ['Face3'])
sketch003.MapMode = 'FlatFace'
doc.recompute()

# Top hole (0, 40)
sketch003.addGeometry(Part.Circle(App.Vector(0.0, 40.0, 0.0), App.Vector(0, 0, 1), 4.0), False)
sketch003.addConstraint(Sketcher.Constraint('Diameter', 0, 8.0))
sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 0, 3, -2))
sketch003.addConstraint(Sketcher.Constraint('DistanceY', -1, 1, 0, 3, 40.0))

# Right hole (40, 0)
sketch003.addGeometry(Part.Circle(App.Vector(40.0, 0.0, 0.0), App.Vector(0, 0, 1), 4.0), False)
sketch003.addConstraint(Sketcher.Constraint('Diameter', 1, 8.0))
sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 1, 3, -1))
sketch003.addConstraint(Sketcher.Constraint('DistanceX', -1, 1, 1, 3, 40.0))

# Bottom hole (0, -40)
sketch003.addGeometry(Part.Circle(App.Vector(0.0, -40.0, 0.0), App.Vector(0, 0, 1), 4.0), False)
sketch003.addConstraint(Sketcher.Constraint('Diameter', 2, 8.0))
sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 2, 3, -2))
sketch003.addConstraint(Sketcher.Constraint('DistanceY', 2, 3, -1, 1, 40.0))

# Left hole (-40, 0)
sketch003.addGeometry(Part.Circle(App.Vector(-40.0, 0.0, 0.0), App.Vector(0, 0, 1), 4.0), False)
sketch003.addConstraint(Sketcher.Constraint('Diameter', 3, 8.0))
sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 3, 3, -1))
sketch003.addConstraint(Sketcher.Constraint('DistanceX', 3, 3, -1, 1, 40.0))

doc.recompute()

pocket001 = body.newObject('PartDesign::Pocket', 'Pocket001')
pocket001.Profile = (sketch003, [''])
pocket001.Length = 15.0
pocket001.TaperAngle = 0.0
pocket001.UseCustomVector = False
pocket001.Direction = (0, 0, -1)
pocket001.ReferenceAxis = (sketch003, ['N_Axis'])
pocket001.AlongSketchNormal = True
pocket001.Type = 0
pocket001.Midplane = False
pocket001.Reversed = False
pocket001.Offset = 0.0
sketch003.Visibility = False
pad001.Visibility = False
doc.recompute()

# ---------------------------------------------------------------
# Fillet chain
# ---------------------------------------------------------------
fillet = body.newObject('PartDesign::Fillet', 'Fillet')
fillet.Base = (pocket001, ['Edge11'])
fillet.Radius = 5.0
pocket001.Visibility = False
doc.recompute()

fillet001 = body.newObject('PartDesign::Fillet', 'Fillet001')
fillet001.Base = (fillet, ['Edge2'])
fillet001.Radius = 5.0
fillet.Visibility = False
doc.recompute()

fillet002 = body.newObject('PartDesign::Fillet', 'Fillet002')
fillet002.Base = (fillet001, ['Edge20'])
fillet002.Radius = 5.0
fillet.Visibility = False
doc.recompute()

fillet003 = body.newObject('PartDesign::Fillet', 'Fillet003')
fillet003.Base = (fillet002, ['Edge29'])
fillet003.Radius = 2.0
fillet.Visibility = False
doc.recompute()

fillet004 = body.newObject('PartDesign::Fillet', 'Fillet004')
fillet004.Base = (fillet003, ['Edge8'])
fillet004.Radius = 2.0
fillet.Visibility = False
doc.recompute()

fillet005 = body.newObject('PartDesign::Fillet', 'Fillet005')
fillet005.Base = (fillet004, ['Edge12'])
fillet005.Radius = 5.0
fillet.Visibility = False
doc.recompute()

doc.recompute()

# ---------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------
body.Tip = fillet005
fillet005.Visibility = True
doc.recompute()

# Optional: save the file
# doc.saveAs(u"C:/path/to/your/flange_bolted.FCStd")