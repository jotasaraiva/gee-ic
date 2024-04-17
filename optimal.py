import ee
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# inicializa conversão automática de objetos Python/R
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

# inicializa Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# região e intervalo de interessse
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

# coleta de imagens
sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
    .filterBounds(geom)\
    .filterDate(inicio,fim)\
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    
v_emit_asc = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
v_emit_desc = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

# funções de amplitude e transformação em DF
def add_amplitude(image, VV = "VV", VH = "VH"):
    amplitude = image\
        .expression('(VV ** 2 + VH ** 2) ** (1 / 2)', {'VV':image.select(VV), 'VH':image.select(VH)})\
        .rename('amplitude')
    return image.addBands(amplitude)

def ee_to_pandas(imagem, geometria, bandas, scale):
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

# funções para reordenar cronologicamente
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

# extraindo um pixel em série temporal
scale = StandardScaler()

x = reorder\
    .drop(['latitude','longitude'], axis=1)\
    .iloc[0,:]

astsa = importr('astsa')
base = importr('base')
stats = importr('stats')

x = base.scale(np.reshape(np.array(x), (-1, 1)))

# função de envelope espectral
def get_specenv(x, *args): 
    scaled_time_series = stats.ts(robjects.FloatVector(list(x))).ravel()
    arrays = [i(scaled_time_series) for i in args]
    arrays.insert(0, scaled_time_series)
    arrays = np.stack(arrays, axis=1)

    spec_env = astsa.specenv(arrays, real=True, plot=False)
    num_coefs = spec_env[:,2:].shape[1]
    names_col = ['frequency', 'specenv'] + [f"coef{i}" for i in range(1, num_coefs+1)]
    
    return pd.DataFrame(spec_env, columns=names_col)

spec_env = get_specenv(x, np.abs, np.square)

# extrair melhores coeficientes
beta = spec_env[spec_env.specenv == spec_env.specenv.max()].iloc[:,2:].squeeze()

# escalar coeficientes
b = beta

# função de otimização
opt = lambda x: b[0]*x + b[1]*np.abs(x) + b[2]*np.square(x)

# resultado da série otimizada
res = np.concatenate([x, opt(x)], axis = 1)

df_res = pd.DataFrame(res, columns = ['original', 'opt'])

sbn.lineplot(df_res)

names = list(reorder.drop(['latitude','longitude'], axis=1).columns)

scaled = scale.fit_transform(reorder.drop(['latitude','longitude'], axis=1).values.T).T

scaled_df = pd.DataFrame(scaled, columns=names)

def optimize(x, *args):
    if type(x) == pd.core.series.Series:
        x = np.array(x)
    arr = list(x.flatten())
    arrays = [arr] + [list(i(arr)) for i in args]
    mat = np.array(arrays).T
    spec_env = astsa.specenv(mat, real=True, plot=False)
    beta = spec_env[spec_env[:,1]==max(spec_env[:,1]), 2:].ravel()
    opt = lambda l: np.array([l] + [list(k(l)) for k in args]).T * beta
    return opt(arr).sum(axis=1).reshape(-1,1)

final_test = np.concatenate(
    (np.array(scaled_df.iloc[0,:]).reshape(-1,1),
    optimize(scaled_df.iloc[0,:], np.abs, np.square),
    optimize(x, np.abs, np.square)),
    axis=1
)

sbn.lineplot(final_test)

startTime = datetime.now()
print(datetime.now() - startTime)
novo_df = scaled_df.apply(lambda x: optimize(x, np.abs, np.square), axis=1)
print(datetime.now() - startTime)

