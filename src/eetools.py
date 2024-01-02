import ee
import numpy as np
import pandas as pd

def ndvi(image, red = "B4", nir = "B5"):
    ndvi = image.expression('(nir - red)/(nir + red)', {'nir':image.select(nir), "red":image.select(red)}).rename('ndvi')
    
    return image.addBands(ndvi)

def mascara_agua(imagem):
    qa = imagem.select('pixel_qa')
    
    return qa.bitwiseAnd(1 << 2).eq(0)

def mascara_nuvem(imagem):
    qa = imagem.select('pixel_qa')
    
    return qa.bitwiseAnd(1 << 3).eq(0) and (qa.bitwiseAnd(1 << 5).eq(0)) and (qa.bitwiseAnd(1 << 6).eq(0)) and (qa.bitwiseAnd(1 << 7).eq(0))

def aplicar_mascara(imagem):
    vazio = ee.Image(99999)
    agua = vazio.updateMask(mascara_agua(imagem).Not()).rename('agua')
    nuvem = vazio.updateMask(mascara_nuvem(imagem).Not()).rename('nuvem')
    sem_nuvem = vazio.updateMask(mascara_nuvem(imagem)).rename('sem_nuvem')
    ndvi = imagem.expression('(nir - red) / (nir + red)',{'nir':imagem.select('B5'),'red':imagem.select('B4')}).rename('ndvi')
    
    return imagem.addBands([ndvi,agua,nuvem,sem_nuvem])

def aplicar_mascara_banda(imagem, banda_mascara, banda_origem, banda_destino):
    imagem_mascara = imagem\
        .select(banda_origem)\
        .updateMask(imagem.select(banda_mascara))\
        .rename(banda_destino)
    imagem_mascara = ee.Image(99999)\
        .blend(imagem_mascara)\
        .rename(banda_destino)
    
    return imagem.addBands([imagem_mascara])

def extrair_latlong_pixel(imagem, geometria, bandas):
    imagem = imagem.addBands(ee.Image.pixelLonLat())
    coordenadas = imagem.select(['longitude','latitude'] + bandas)\
        .reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=geometria,
            scale=30,
            bestEffort=True
        )
    bandas_valores = []
    for banda in bandas:
        bandas_valores.append(np.array(ee.List(coordenadas.get(banda)).getInfo()).astype(float))
    
    return np.array(ee.List(coordenadas.get('latitude')).getInfo()).astype(float), np.array(ee.List(coordenadas.get('longitude')).getInfo()).astype(float), bandas_valores

def extrair_lonlat(imagem, geometria, bandas, scale=30):
    imagem = imagem.addBands(ee.Image.pixelLonLat())
    
    coordenadas = imagem.select(["longitude","latitude"] + bandas)\
        .reduceRegion(reducer=ee.Reducer.toList(),
                     geometry=geometria,
                     scale=scale,
                     bestEffort=True)
    
    coordenadas = coordenadas.getInfo()
    
    return pd.DataFrame.from_dict(coordenadas)