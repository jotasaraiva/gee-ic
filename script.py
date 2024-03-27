import ee
import geemap
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import saveModule
from osgeo import gdal
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects

rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()
    
    
coordenadas = "-48.534679,-22.508117,-48.50481,-22.538879"
x1, y1, x2, y2 = coordenadas.split(",")

datas = "2020-01-01,2020-12-31"
inicio, fim = datas.split(",")
escala = 30
dummy_value = 99999

geom = ee.Geometry.Polygon([[[float(x1),float(y2)],
                             [float(x2),float(y2)],
                             [float(x2),float(y1)],
                             [float(x1),float(y1)],
                             [float(x1),float(y2)]]])

latitude_central = (float(x1)+float(x2))/2
longitude_central = (float(y1)+float(y2))/2

sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
    .filterBounds(geom)\
    .filterDate(inicio,fim)\
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    
v_emit_asc = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
v_emit_desc = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

my_map = geemap.Map(center=(longitude_central, latitude_central), zoom=11)
my_map.addLayer(geom)

rgb = ee.Image.rgb(
    v_emit_desc.mean().select('VV'),
    v_emit_desc.mean().select('VH'),
    v_emit_desc.mean().select('VV').divide(v_emit_desc.mean().select('VH'))
).clip(geom)

my_map.addLayer(rgb, {'min': [-25,-25,0], 'max': [0,0,2]})

my_map.to_html(filename=os.path.join(os.getcwd(), 'my_map.html'))

def add_amplitude(image, VV = "VV", VH = "VH"):
    amplitude = image\
        .expression('(VV ** 2 + VH ** 2) ** (1 / 2)', {'VV':image.select(VV), 'VH':image.select(VH)})\
        .rename('amplitude')
    return image.addBands(amplitude)

def ee_to_pandas(imagem, geometria, bandas, scale=30):
    imagem = imagem.addBands(ee.Image.pixelLonLat())
    
    coordenadas = imagem.select(["longitude","latitude"] + bandas)\
        .reduceRegion(reducer=ee.Reducer.toList(),
                     geometry=geometria,
                     scale=scale,
                     bestEffort=True)
    
    coordenadas = coordenadas.getInfo()
    
    return pd.DataFrame.from_dict(coordenadas)

image = ee.Image(dummy_value).blend(v_emit_desc.map(add_amplitude).select('amplitude').toBands())
image_names = image.bandNames().getInfo()

df = ee_to_pandas(image, geom, image_names, scale=10)

export = df.iloc[:,[0,1,2,3,4,5,6,31,32]]
export_columns = list(export.drop(['latitude','longitude'], axis = 1).columns)

saveModule.save_tiff_from_df(
    export,
    export_columns,
    99999,
    "assets/amplitude.tif",
    "EPSG:4326"   
)

filepath = r"assets/amplitude.tif"
raster = gdal.Open(filepath)
array = raster.GetRasterBand(5).ReadAsArray()
plt.imshow(array)

astsa = importr('astsa')
base = importr('base')
stats = importr('stats')

def get_specenv(x, *args): 
    scaled_time_series = stats.ts(robjects.FloatVector(list(x))).ravel()
    arrays = [i(scaled_time_series) for i in args]
    arrays.insert(0, scaled_time_series)
    arrays = np.stack(arrays, axis=1)

    spec_env = astsa.specenv(arrays, real=True, plot=False)
    num_coefs = spec_env[:,2:].shape[1]
    names_col = ['frequency', 'specenv'] + [f"coef{i}" for i in range(1, num_coefs+1)]
    
    return pd.DataFrame(spec_env, columns=names_col)