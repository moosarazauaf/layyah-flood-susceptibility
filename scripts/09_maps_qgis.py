"""
Publication maps, rendered by QGIS 3.40, one file each at 600 dpi.

    fig1_study_area.png     seven districts, Indus corridor, Layyah held out
    fig2_flood_record.png   Layyah: monsoons flooded 2000-2019 (of 20), and
                            land first flooded 2020-2021
    fig6_susceptibility.png Layyah: susceptibility, quintile classes

Run headless:
    "C:\\Program Files\\QGIS 3.40.8\\bin\\python-qgis-ltr.bat" scripts\\09_maps_qgis.py

Before drawing, the strips stitched from Earth Engine are checked against
outputs/layyah_map_stats.json: each class should hold about a fifth of the
corridor, and about 87% of 2000-2019 flooding should fall in classes 4-5.
"""
import json
import os
import sys

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsApplication, QgsColorRampShader, QgsFillSymbol, QgsHueSaturationFilter,
    QgsLayoutItemLabel, QgsLayoutPoint, QgsLayoutSize, QgsPalLayerSettings,
    QgsProject, QgsRasterLayer, QgsRasterRange, QgsRasterShader, QgsRectangle,
    QgsSingleBandPseudoColorRenderer, QgsSingleSymbolRenderer,
    QgsTextBufferSettings, QgsTextFormat, QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import qgis_layout as L  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "paper", "figures")
TEHSILS = os.path.join(ROOT, "data", "layyah_tehsils.geojson")
PUNJAB = os.path.join(ROOT, "data", "punjab_districts.geojson")
S2 = os.path.join(OUT, "layyah_s2_truecolour_50m.tif")
TRAIN = ["Mianwali District", "Bhakkar District", "Muzaffargarh District",
         "Dera Ghazi Khan District", "Rajanpur District", "Rahim Yar Khan District"]
CLASS_COL = ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c", "#800026"]
FLOOD_COL = ["#c6dbef", "#6baed6", "#2171b5", "#08519c", "#08306b"]
gdal.UseExceptions()


def stitch(folder, name):
    """Mosaic the strips and cut them to Layyah's GAUL polygon.

    Earth Engine writes pixels outside a clip's footprint as 0 (tagged
    nodata) even after unmask(255), and 0 is also a valid low score inside
    the corridor, so the value cannot tell outside from very low. The
    polygon can: everything outside it becomes 255. It is the same boundary
    the Earth Engine statistics used."""
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif"))
    mosaic = os.path.join(OUT, name + ".vrt")
    gdal.BuildVRT(mosaic, files, srcNodata=255, VRTNodata=255)
    path = os.path.join(OUT, name + ".tif")
    gdal.Warp(path, mosaic, cutlineDSName=os.path.join(ROOT, "data", "layyah_district.geojson"),
              srcNodata=255, dstNodata=255, creationOptions=["COMPRESS=LZW"])
    return path


def check(score_vrt, inv_vrt, cuts):
    s = gdal.Open(score_vrt)
    a = s.ReadAsArray().astype(float)
    inside = a != 255
    cls = np.digitize(a[inside] / 100.0, cuts) + 1
    shares = [round(100 * float((cls == c).mean()), 1) for c in range(1, 6)]
    # flooding on the score grid
    g = s.GetGeoTransform()
    inv = gdal.Warp("", inv_vrt, format="MEM", outputBounds=(
        g[0], g[3] + g[5] * s.RasterYSize, g[0] + g[1] * s.RasterXSize, g[3]),
        width=s.RasterXSize, height=s.RasterYSize, resampleAlg="near",
        srcNodata=255, dstNodata=255)
    yrs = inv.GetRasterBand(1).ReadAsArray()
    fl = (yrs >= 1) & (yrs != 255) & inside
    c_fl = np.digitize(a[fl] / 100.0, cuts) + 1
    hi = round(100 * float((c_fl >= 4).mean()), 1)
    print(f"  class shares of corridor (%): {shares}  [expect ~20 each]")
    print(f"  2000-2019 flooding in classes 4-5: {hi}%  [Earth Engine: 87.0%]")
    assert all(15 <= x <= 25 for x in shares), "class shares off: stitching or cuts wrong"
    assert abs(hi - 87.0) < 3, "flooding share disagrees with Earth Engine statistics"


def raster(p, path, name, band, items, discrete=True, nodata=(255,), opacity=1.0):
    lyr = QgsRasterLayer(path, name); assert lyr.isValid(), path
    lyr.dataProvider().setUserNoDataValue(band, [QgsRasterRange(v, v) for v in nodata])
    sh = QgsColorRampShader()
    sh.setColorRampType(QgsColorRampShader.Discrete if discrete else QgsColorRampShader.Exact)
    sh.setColorRampItemList([QgsColorRampShader.ColorRampItem(v, QColor(c), lab)
                             for v, c, lab in items])
    rs = QgsRasterShader(); rs.setRasterShaderFunction(sh)
    r = QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), band, rs)
    r.setOpacity(opacity); lyr.setRenderer(r); p.addMapLayer(lyr)
    return lyr


def vector(p, path, name, fill, outline="#1a1a1a", width="0.4", subset=None, label=None, size=10):
    lyr = QgsVectorLayer(path, name, "ogr"); assert lyr.isValid(), path
    if subset:
        lyr.setSubsetString(subset)
    lyr.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
        {"color": fill, "outline_color": outline, "outline_width": width})))
    if label:
        ls = QgsPalLayerSettings(); ls.fieldName, ls.isExpression = label, True
        ls.centroidWhole = False
        tf = QgsTextFormat(); tf.setFont(L.font(size)); tf.setSize(size)
        b = QgsTextBufferSettings(); b.setEnabled(True); b.setSize(1.0)
        b.setColor(QColor("white")); tf.setBuffer(b); ls.setFormat(tf)
        lyr.setLabeling(QgsVectorLayerSimpleLabeling(ls)); lyr.setLabelsEnabled(True)
    p.addMapLayer(lyr)
    return lyr


def grey_base(p):
    s2 = QgsRasterLayer(S2, "Sentinel-2"); assert s2.isValid()
    s2.hueSaturationFilter().setGrayscaleMode(QgsHueSaturationFilter.GrayscaleLuminosity)
    s2.brightnessFilter().setBrightness(45); s2.brightnessFilter().setContrast(-35)
    p.addMapLayer(s2)
    return s2


def valid_extent(p, path, margin_m=2500):
    ds = gdal.Open(path); a = ds.ReadAsArray(); g = ds.GetGeoTransform()
    if a.ndim == 3:
        a = a[0]
    rows, cols = np.nonzero(a != 255)
    r = QgsRectangle(g[0] + cols.min() * g[1], g[3] + (rows.max() + 1) * g[5],
                     g[0] + (cols.max() + 1) * g[1], g[3] + rows.min() * g[5])
    e = L.to_utm(r, p); e.grow(margin_m)
    return e


def corridor_layout(p, name, layers, legend_order, legend_labels, ext, hide=()):
    """Portrait layout for Layyah's corridor: tall map, legend, scale, locator."""
    W, H = 165, 228
    lay = L.new_layout(p, name, W, H)
    mh = H - 18
    mw = round(mh * ext.width() / ext.height(), 1)
    assert mw < 112, f"map too wide ({mw} mm)"
    m = L.add_map(lay, layers, ext, 11, 8, mw, mh)
    L.add_grid(m, 0.1, 0.1)
    L.add_north(lay, 11 + mw - 10, 11)
    x2 = 11 + mw + 8
    L.add_legend(lay, m, legend_order, legend_labels, x2, 8, W - x2 - 3, hide_titles=hide)
    sb = L.add_scalebar(lay, m, x2, 92, 5, 2); sb.setBackgroundEnabled(False)
    dist = QgsVectorLayer(TEHSILS, "district", "ogr")
    dist.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
        {"color": "#e8e8e8", "outline_color": "#555555", "outline_width": "0.2"})))
    p.addMapLayer(dist)
    d_ext = L.to_utm(dist.extent(), p); d_ext.grow(3000)
    lw = min(W - x2 - 3, 62)
    loc = L.add_map(lay, [dist], d_ext, x2, 110, lw, lw * 0.9, frame=0.3)
    ov = loc.overview(); ov.setLinkedMap(m)
    ov.setFrameSymbol(QgsFillSymbol.createSimple(
        {"color": "0,0,0,0", "outline_color": "#d7301f", "outline_width": "0.6"}))
    return lay


def fig1():
    p = QgsProject(); p.setCrs(L.UTM)
    names = ", ".join(f"'{n}'" for n in TRAIN)
    punjab = vector(p, PUNJAB, "Punjab districts", "#f2f2f2", "#ffffff", "0.3")
    train = vector(p, PUNJAB, "Training districts (six)", "#c6dbef", "#4d4d4d", "0.3",
                   subset=f"\"ADM2_NAME\" IN ({names})",
                   label="replace(\"ADM2_NAME\", ' District', '')", size=9)
    test = vector(p, PUNJAB, "Test district: Layyah", "#fdae6b", "#4d4d4d", "0.4",
                  subset="\"ADM2_NAME\" = 'Layyah District'",
                  label="'Layyah'", size=10)
    ctx = os.path.join(OUT, "context_250m.tif")
    corr = raster(p, ctx, "Indus corridor (15 km)", 1,
                  [(1, "#08519c", "Indus floodplain corridor (15 km)")],
                  discrete=False, nodata=(0,), opacity=0.45)
    indus = raster(p, ctx, "Indus main stem", 2, [(1, "#08306b", "Indus main stem")],
                   discrete=False, nodata=(0,))
    sel = QgsVectorLayer(PUNJAB, "sel", "ogr")
    sel.setSubsetString(f"\"ADM2_NAME\" IN ({names}, 'Layyah District')")
    ext = L.to_utm(sel.extent(), p); ext.grow(15000)
    W, H = 165, 228
    lay = L.new_layout(p, "fig1", W, H)
    mh = H - 18
    mw = round(mh * ext.width() / ext.height(), 1)
    m = L.add_map(lay, [indus, corr, test, train, punjab], ext, 11, 8, mw, mh)
    L.add_grid(m, 1.0, 1.0)
    L.add_north(lay, 11 + mw - 10, 11)
    x2 = 11 + mw + 8
    L.add_legend(lay, m, [train, test, corr, indus],
                 {"Indus corridor (15 km)": {0: "Indus floodplain|corridor (15 km)"}},
                 x2, 8, W - x2 - 3,
                 hide_titles=("Indus corridor (15 km)", "Indus main stem"))
    sb = L.add_scalebar(lay, m, x2, 62, 50, 2); sb.setBackgroundEnabled(False)
    allp = QgsVectorLayer(PUNJAB, "Punjab", "ogr")
    allp.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
        {"color": "#e3e3e3", "outline_color": "#ffffff", "outline_width": "0.15"})))
    p.addMapLayer(allp)
    pj = L.to_utm(allp.extent(), p); pj.grow(15000)
    lw = W - x2 - 3
    loc = L.add_map(lay, [allp], pj, x2, 82, lw, lw * 1.05, frame=0.3)
    ov = loc.overview(); ov.setLinkedMap(m)
    ov.setFrameSymbol(QgsFillSymbol.createSimple(
        {"color": "0,0,0,0", "outline_color": "#d7301f", "outline_width": "0.6"}))
    t = QgsLayoutItemLabel(lay); lay.addLayoutItem(t)
    t.setText("Punjab, Pakistan"); t.setFont(L.font(9.5))
    t.attemptMove(QgsLayoutPoint(x2, 82 + lw * 1.05 + 1.5, L.MM))
    t.attemptResize(QgsLayoutSize(lw, 5, L.MM))
    L.export(lay, os.path.join(FIG, "fig1_study_area.png"))


def fig2(inv_vrt, ext):
    p = QgsProject(); p.setCrs(L.UTM)
    base = grey_base(p)
    yrs = raster(p, inv_vrt, "Monsoons flooded, 2000-2019 (of 20)", 1,
                 [(1, FLOOD_COL[0], "1"), (3, FLOOD_COL[1], "2-3"), (6, FLOOD_COL[2], "4-6"),
                  (10, FLOOD_COL[3], "7-10"), (20, FLOOD_COL[4], "11-20")], nodata=(0, 255))
    new = raster(p, inv_vrt, "First flooded 2020-2021", 2,
                 [(1, "#e7298a", "First flooded 2020-2021")], discrete=False, nodata=(0, 255))
    teh = vector(p, TEHSILS, "Tehsil boundary", "0,0,0,0",
                 subset="\"adm3_name\" <> 'Choubara'", label="\"adm3_name\"")
    lay = corridor_layout(p, "fig2", [teh, new, yrs, base], [yrs, new, teh], {}, ext,
                          hide=("First flooded 2020-2021",))
    L.export(lay, os.path.join(FIG, "fig2_flood_record.png"))


def fig6(score_vrt, cuts, ext):
    p = QgsProject(); p.setCrs(L.UTM)
    base = grey_base(p)
    labs = ["1 (lowest)", "2", "3", "4", "5 (highest)"]
    ups = [c * 100 for c in cuts] + [100]
    sc = raster(p, score_vrt, "Flood susceptibility class", 1,
                [(u, c, l) for u, c, l in zip(ups, CLASS_COL, labs)])
    teh = vector(p, TEHSILS, "Tehsil boundary", "0,0,0,0",
                 subset="\"adm3_name\" <> 'Choubara'", label="\"adm3_name\"")
    lay = corridor_layout(p, "fig6", [teh, sc, base], [sc, teh], {}, ext)
    L.export(lay, os.path.join(FIG, "fig6_susceptibility.png"))


if __name__ == "__main__":
    app = None
    if QgsApplication.instance() is None:
        # native platform: the Qt "offscreen" platform has no fonts on Windows
        QgsApplication.setPrefixPath(L.QGIS_PREFIX, True)
        app = QgsApplication([], True); app.initQgis()
    os.makedirs(FIG, exist_ok=True)
    cuts = json.load(open(os.path.join(OUT, "layyah_map_stats.json")))["quintile_cuts"]
    score_vrt = stitch(os.path.join(OUT, "score_strips"), "layyah_score")
    inv_vrt = stitch(os.path.join(OUT, "inventory_strips"), "layyah_inventory")
    check(score_vrt, inv_vrt, cuts)
    ext = valid_extent(QgsProject(), score_vrt)
    fig1()
    fig2(inv_vrt, ext)
    fig6(score_vrt, cuts, ext)
    if app is not None:
        app.exitQgis()
