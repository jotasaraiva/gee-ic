import ee
import re
import pandas as pd

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

def extract_date_string(input_string):
    pattern = r'(\d{8})(?=[a-zA-Z])'
    input_string.replace("_", " ")
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        return None

def rename_geodf(df):
    coords = df.loc[:,['longitude', 'latitude']]
    bands = df.drop(['longitude', 'latitude'], axis=1)
    bands.columns = list(map(extract_date_string, list(bands.columns)))
    new_df = bands.join(coords)
    return new_df

renamed_df = rename_geodf(df)

reorder = renamed_df.reindex(sorted(renamed_df.columns), axis=1)

reorder.columns
