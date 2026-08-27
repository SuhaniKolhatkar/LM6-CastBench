# -*- coding: utf-8 -*-
"""
FreeCAD macro - recreates 'brake.FCStd'
PartDesign Body > Sketch (XY plane) > Pad (8 mm)
"""

import FreeCAD as App
import Part
import Sketcher

doc = App.newDocument("brake")

# ---------------------------------------------------------------
# Body
# ---------------------------------------------------------------
body = doc.addObject('PartDesign::Body', 'Body')

# ---------------------------------------------------------------
# Sketch on XY plane
# ---------------------------------------------------------------
sketch = doc.addObject('Sketcher::SketchObject', 'Sketch')
body.addObject(sketch)
sketch.AttachmentSupport = [(body.Origin.OriginFeatures[3], '')]  # XY_Plane
sketch.MapMode = 'FlatFace'

# ---------------------------------------------------------------
# Add geometry one at a time (index order matters for constraints)
# ---------------------------------------------------------------

# idx0 - Hole 1: R7 (Ø14) circle, left-lower hub
sketch.addGeometry(Part.Circle(App.Vector(-82.5587518926589752, 0.0, 0.0),
                                App.Vector(0, 0, 1), 7.0), False)

# idx1 - Hole 2: R6 (Ø12) circle, left-upper hub
sketch.addGeometry(Part.Circle(App.Vector(-84.2490282719526107, 23.0, 0.0),
                                App.Vector(0, 0, 1), 6.0), False)

# idx2 - R10.5 outer arc of left-lower hub
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(-82.5587518926589752, 0.0, 0.0),
                App.Vector(0, 0, 1), 10.5),
    2.6748623510764644, 6.0188533385699472), False)

# idx3 - connecting line
sketch.addGeometry(Part.LineSegment(
    App.Vector(-72.4234457767213229, -2.7432772255528048, 0.0),
    App.Vector(-70.5534731731366662,  4.1655195191612426, 0.0)), False)

# idx4 - R9.122 fillet arc
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(-61.7482540794578938, 1.7822509005935225, 0.0),
                App.Vector(0, 0, 1), 9.1220530910500024),
    1.1728587767723522, 2.8772606849801536), False)

# idx5 - lower Bezier curve (4 poles, cubic) -> converted to BSpline for Sketcher
bez1 = Part.BezierCurve()
bez1.setPoles([
    App.Vector(-58.2132954664423607, 10.1915267433877297, 0.0),
    App.Vector(-30.6504116037719676, -1.3949232115608843, 0.0),
    App.Vector(-3.8034599096136952,  -7.1031023045212240, 0.0),
    App.Vector(18.9413175535375089,  -4.0799138391747398, 0.0)])
sketch.addGeometry(bez1.toBSpline(), False)

# idx6 - small R2.452 fillet arc near right end
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(19.2644399303790657, -6.5109057005383768, 0.0),
                App.Vector(0, 0, 1), 2.4523722189814454),
    0.6724593155872700, 1.7029396777116752), False)

# idx7 - R8 (Ø16) arc, right end
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(27.4412481073410284, 0.0, 0.0),
                App.Vector(0, 0, 1), 8.0),
    3.8140519691770636, 9.4090425313055768), False)

# idx8 - tiny R0.505 fillet arc near right end
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(18.9373426947153547, 0.1338236490482382, 0.0),
                App.Vector(0, 0, 1), 0.5049583206461855),
    4.8135326571618791, 6.2674498777157828), False)

# idx9 - upper Bezier curve (4 poles, cubic) -> converted to BSpline for Sketcher
bez2 = Part.BezierCurve()
bez2.setPoles([
    App.Vector(-52.2345052606587004, 24.0393897989436489, 0.0),
    App.Vector(-33.8050054836408691,  7.4890195502119203, 0.0),
    App.Vector(-8.3005575315068434,  -3.1381029682104540, 0.0),
    App.Vector(18.9883290000000038,  -0.3685540000000018, 0.0)])
sketch.addGeometry(bez2.toBSpline(), False)

# idx10 - R11.178 fillet arc near left-upper hub
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(-60.8646701369499183, 16.9345836523270776, 0.0),
                App.Vector(0, 0, 1), 11.1784621559931576),
    0.6887594529619723, 1.3226286989555898), False)

# idx11 - connecting line, upper hub to spline
sketch.addGeometry(Part.LineSegment(
    App.Vector(-82.2093480389212488, 32.7897755105508537, 0.0),
    App.Vector(-58.1189253057287800, 27.7705839058705237, 0.0)), False)

# idx12 - R10 outer arc of left-upper hub
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(-84.2490282719526107, 23.0, 0.0),
                App.Vector(0, 0, 1), 10.0),
    1.3653868727566090, 3.9194934936138539), False)

# idx13 - small R4.312 fillet arc, left side
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(-94.4448393207024566, 12.9559366776967071, 0.0),
                App.Vector(0, 0, 1), 4.3121546583401518),
    0.0, 0.7779008400240612), False)

# idx14 - short vertical connecting line, left side
sketch.addGeometry(Part.LineSegment(
    App.Vector(-90.1326846623623084, 12.9559366776967053, 0.0),
    App.Vector(-90.1326846623623084, 10.1370663681643087, 0.0)), False)

# idx15 - small R4.097 fillet arc, left side
sketch.addGeometry(Part.ArcOfCircle(
    Part.Circle(App.Vector(-94.2296158889424618, 10.1370663681643105, 0.0),
                App.Vector(0, 0, 1), 4.0969312265801410),
    5.9129335844203599, 6.2831853071795862), False)

# idx16 - connecting line back to start
sketch.addGeometry(Part.LineSegment(
    App.Vector(-90.4103079250263306, 8.6545913592368979, 0.0),
    App.Vector(-91.9357172617598195, 4.7246714665343053, 0.0)), False)

# ---------------------------------------------------------------
# Coincident constraints -- stitch the 15 outer-profile edges
# into one closed wire (Pos 1 = start point, Pos 2 = end point)
# ---------------------------------------------------------------
c = sketch.addConstraint
c(Sketcher.Constraint('Coincident', 2, 2, 3, 1))
c(Sketcher.Constraint('Coincident', 3, 2, 4, 2))
c(Sketcher.Constraint('Coincident', 4, 1, 5, 1))
c(Sketcher.Constraint('Coincident', 5, 2, 6, 2))
c(Sketcher.Constraint('Coincident', 6, 1, 7, 1))
c(Sketcher.Constraint('Coincident', 7, 2, 8, 2))
c(Sketcher.Constraint('Coincident', 8, 1, 9, 2))
c(Sketcher.Constraint('Coincident', 9, 1, 10, 1))
c(Sketcher.Constraint('Coincident', 10, 2, 11, 2))
c(Sketcher.Constraint('Coincident', 11, 1, 12, 1))
c(Sketcher.Constraint('Coincident', 12, 2, 13, 2))
c(Sketcher.Constraint('Coincident', 13, 1, 14, 1))
c(Sketcher.Constraint('Coincident', 14, 2, 15, 2))
c(Sketcher.Constraint('Coincident', 15, 1, 16, 1))
c(Sketcher.Constraint('Coincident', 16, 2, 2, 1))   # closes the loop

doc.recompute()

# ---------------------------------------------------------------
# Pad
# ---------------------------------------------------------------
pad = body.newObject('PartDesign::Pad', 'Pad')
pad.Profile = sketch
pad.Length = 8.0
pad.Midplane = False
pad.Reversed = False
pad.Type = 'Length'

sketch.Visibility = False
doc.recompute()

try:
    import FreeCADGui as Gui
    Gui.ActiveDocument.ActiveView.viewAxonometric()
    Gui.SendMsgToActiveView("ViewFit")
except Exception:
    pass

doc.saveAs(u"brake_recreated.FCStd")