"""
QGIS print-layout helpers shared by the map scripts (QGIS 3.40).

Lifted from the earlier Layyah radar study, where each was debugged:
graticule sides pinned to one coordinate each (UTM tilts the graticule),
legend user labels keyed by the original node index, "|" as the legend
wrap string, and fonts set explicitly. Layouts are 165 mm wide, the
journal's page width, so sizes here are printed sizes (minimum 9 pt).
"""
import os

from qgis.core import (
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsLayoutExporter,
    QgsLayoutItemLegend, QgsLayoutItemMap, QgsLayoutItemMapGrid,
    QgsLayoutItemPicture, QgsLayoutItemScaleBar, QgsLayoutMeasurement,
    QgsLayoutPoint, QgsLayoutSize, QgsLegendRenderer, QgsLegendStyle,
    QgsMapLayerLegendUtils, QgsPrintLayout, QgsUnitTypes,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont

QGIS_PREFIX = r"C:/Program Files/QGIS 3.40.8/apps/qgis-ltr"
NORTH_SVG = QGIS_PREFIX + "/svg/arrows/NorthArrow_02.svg"
UTM = QgsCoordinateReferenceSystem("EPSG:32642")
WGS = QgsCoordinateReferenceSystem("EPSG:4326")
TNR = "Times New Roman"
MM = QgsUnitTypes.LayoutMillimeters
DPI = 600


def font(size, bold=False):
    f = QFont(TNR); f.setPointSizeF(size); f.setBold(bold); return f


def to_utm(rect, p):
    return QgsCoordinateTransform(WGS, UTM, p.transformContext()).transformBoundingBox(rect)


def new_layout(p, name, w, h):
    lay = QgsPrintLayout(p); lay.initializeDefaults(); lay.setName(name)
    lay.pageCollection().page(0).setPageSize(QgsLayoutSize(w, h, MM))
    return lay


def add_map(lay, layers, extent, x, y, w, h, crs=UTM, frame=0.35):
    m = QgsLayoutItemMap(lay); lay.addLayoutItem(m)
    m.attemptMove(QgsLayoutPoint(x, y, MM)); m.attemptResize(QgsLayoutSize(w, h, MM))
    m.setCrs(crs); m.setLayers(layers); m.zoomToExtent(extent)
    m.setFrameEnabled(True)
    m.setFrameStrokeWidth(QgsLayoutMeasurement(frame, MM))
    m.setBackgroundColor(QColor("white"))
    return m


def add_grid(m, dx, dy):
    g = m.grid()
    g.setEnabled(True); g.setCrs(WGS)
    g.setIntervalX(dx); g.setIntervalY(dy)
    g.setStyle(QgsLayoutItemMapGrid.FrameAnnotationsOnly)
    g.setFrameStyle(QgsLayoutItemMapGrid.InteriorTicks); g.setFrameWidth(1.4)
    g.setAnnotationEnabled(True)
    g.setAnnotationFormat(QgsLayoutItemMapGrid.DecimalWithSuffix)
    g.setAnnotationPrecision(1)
    g.setAnnotationFont(font(9))
    # in UTM the graticule tilts, so a latitude line can cross the bottom
    # edge; pin each side to one coordinate
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.LatitudeOnly, QgsLayoutItemMapGrid.Left)
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.LongitudeOnly, QgsLayoutItemMapGrid.Bottom)
    for side in (QgsLayoutItemMapGrid.Top, QgsLayoutItemMapGrid.Right):
        g.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, side)
    g.setAnnotationDirection(QgsLayoutItemMapGrid.Vertical, QgsLayoutItemMapGrid.Left)
    m.updateBoundingRect()


def add_scalebar(lay, m, x, y, seg_km, nseg):
    sb = QgsLayoutItemScaleBar(lay); lay.addLayoutItem(sb)
    sb.setLinkedMap(m); sb.setStyle("Single Box")
    sb.setUnits(QgsUnitTypes.DistanceKilometers); sb.setUnitLabel("km")
    sb.setUnitsPerSegment(seg_km); sb.setNumberOfSegmentsLeft(0)
    sb.setNumberOfSegments(nseg); sb.setHeight(1.8)
    sb.setFont(font(9))
    sb.setBackgroundEnabled(True); sb.setBackgroundColor(QColor(255, 255, 255, 215))
    sb.attemptMove(QgsLayoutPoint(x, y, MM))
    return sb


def add_north(lay, x, y, size=8):
    na = QgsLayoutItemPicture(lay); lay.addLayoutItem(na)
    na.setPicturePath(NORTH_SVG)
    na.attemptMove(QgsLayoutPoint(x, y, MM)); na.attemptResize(QgsLayoutSize(size, size, MM))


def add_legend(lay, m, order, labels, x, y, w, title="", hide_titles=()):
    """order: layers top to bottom. labels: {layer name: {node index: text}}."""
    lg = QgsLayoutItemLegend(lay); lay.addLayoutItem(lg)
    lg.setLinkedMap(m); lg.setAutoUpdateModel(False); lg.setTitle(title)
    root = lg.model().rootGroup()
    root.removeAllChildren()
    for lyr in order:
        node = root.addLayer(lyr)
        if lyr.name() in hide_titles:
            QgsLegendRenderer.setNodeLegendStyle(node, QgsLegendStyle.Hidden)
        nodes = lg.model().layerLegendNodes(node)
        keep = [i for i, n in enumerate(nodes)
                if not str(n.data(Qt.DisplayRole)).startswith("Band ")]
        QgsMapLayerLegendUtils.setLegendNodeOrder(node, keep)
        # user labels are keyed by the node's original index, not its
        # position after the "Band N" row is dropped
        for pos, text in labels.get(lyr.name(), {}).items():
            QgsMapLayerLegendUtils.setLegendNodeUserLabel(node, keep[pos], text)
        lg.model().refreshLayerLegend(node)
    for style, size, bold in ((QgsLegendStyle.Title, 10, True),
                              (QgsLegendStyle.Subgroup, 10, True),
                              (QgsLegendStyle.Group, 10, True),
                              (QgsLegendStyle.SymbolLabel, 9.5, False)):
        lg.setStyleFont(style, font(size, bold))
    lg.setSymbolWidth(6); lg.setSymbolHeight(3.6)
    lg.setWrapString("|")
    lg.setBackgroundEnabled(False)
    lg.attemptMove(QgsLayoutPoint(x, y, MM))
    lg.setResizeToContents(True)
    lg.adjustBoxSize()
    return lg


def export(lay, path):
    s = QgsLayoutExporter.ImageExportSettings(); s.dpi = DPI
    res = QgsLayoutExporter(lay).exportToImage(path, s)
    assert res == QgsLayoutExporter.Success, f"export failed ({res}): {path}"
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)")
