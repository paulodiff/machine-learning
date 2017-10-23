# Data Augmentation
# Preparaun set di immagini dalle originali con alcune trasformazioni mantenendo la dimensione originale
#
# 1. Rotazione destra sinistra (3°)
# 2. Traslazione 10, 20 px
# 3. Zoom
#

import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps


import skimage
import os
from skimage.viewer import ImageViewer
from skimage import data
from skimage import io
from skimage import transform
from skimage.transform import rescale

SOURCE_FILE_FOLDER = 'C:/Users/M05831/Downloads/C_I/output_png'
SOURCE_FILE_FOLDER_COLLECTION = 'C:/Users/M05831/Downloads/C_I/output_png/*.png'
OUTPUT_FILE_FOLDER = 'C:/Users/M05831/Downloads/C_I/output_test'
print('START')
print(SOURCE_FILE_FOLDER)

def rotate_image_and_save(img2rotate, rotation, fname):
    print('rotata_image:', rotation)
    im2 = img2rotate.convert('RGBA')
    print(im2.size)
    # rotated image
    rot = im2.rotate(rotation, expand=1)
    # a white image same size as rotated image
    # fff = Image.new('RGBA', rot.size, (255,) * 4)
    print(rot.size)
    fff = Image.new('RGBA', (160,120), (255,) * 4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(rot, fff, rot)
    print(out.size)
    out.save(fname, "PNG")




files = [ f for f in os.listdir(SOURCE_FILE_FOLDER) if os.path.isfile(os.path.join(SOURCE_FILE_FOLDER, f))]

for f in files:
    print(f)
    filename = os.path.join(SOURCE_FILE_FOLDER, f)
    print('open:' + filename)
    im = Image.open(filename)
    print(im.size)
    print(im.mode)

    # PIL.ImageOps.grayscale(image)
    imGray = PIL.ImageOps.grayscale(im)
    filename = os.path.join(OUTPUT_FILE_FOLDER, 'G1' + f)
    imGray.save(filename, "PNG")
    print('GrayScale:' + filename)
    print(imGray.size)
    print(imGray.mode)
    print(list(imGray.getdata()))


    # 1 - Espansione con bordo bianco
    imExp = PIL.ImageOps.expand(imGray , 10, '#ffffff')
    filename = os.path.join(OUTPUT_FILE_FOLDER, 'X1' + f)
    print('border:' + filename)
    #imExp.save(filename, "PNG")


    # PIL.ImageOps.fit(image, size, method=0, bleed=0.0, centering=(0.5, 0.5))

    ###imRot = imExp.rotate(3.0)  # degrees counter-clockwise
    filename = os.path.join(OUTPUT_FILE_FOLDER, 'R2' + f)
    print('border:' + filename)
    ###imRot.save(filename, "PNG")

    # ZOOM_1
    # ZOOM_2

    # viewer = ImageViewer(moon)
    # viewer.show()
    rotate_image_and_save(imExp, 3.0, os.path.join(OUTPUT_FILE_FOLDER, 'R1' + f))
    rotate_image_and_save(imExp,-3.0, os.path.join(OUTPUT_FILE_FOLDER, 'R2' + f))
    rotate_image_and_save(imExp, 4.0, os.path.join(OUTPUT_FILE_FOLDER, 'R3' + f))
    rotate_image_and_save(imExp,-4.0, os.path.join(OUTPUT_FILE_FOLDER, 'R4' + f))

#ic = io.ImageCollection(SOURCE_FILE_FOLDER_COLLECTION, conserve_memory=False, plugin='test')
#io.imshow_collection(ic)


print('END')
#filename = os.path.join(skimage.data_dir, 'moon.png')
#moon = io.imread(filename)
#camera = data.camera()
#type(camera)

#image = data.coins()
#viewer = ImageViewer(image)
#viewer.show()