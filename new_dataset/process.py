import sys
sys.path.append('/opt/ASAP/bin')
import multiresolutionimageinterface as mir

from PIL import Image
import os 

img_list = os.listdir('/disk2/Train_Tumor/')

xml_root = ['/disk2/Ground_Truth/XML/'+n[0:-4]+'.xml' for n in img_list]
mask_root = ['/disk2/Ground_Truth/Mask/'+n[0:-4]+'_Mask.tif' for n in img_list]
img_root = ['/disk2/Train_Tumor/'+n for n in img_list]

for i in range(0, len(img_list)):
    mr_image = reader.open('/disk2/Train_Tumor/Tumor_001.tif')
    mr_image = reader.open('/disk2/Ground_Truth/Mask/Tumor_001_Mask.tif')

def create_dataset(img_name):
    
    reader = mir.MultiResolutionImageReader()
    
    xml_root = '/disk2/Ground_Truth/XML/'+img_name+'.xml'
    mask_root = '/disk2/Ground_Truth/Mask/'+img_name+'_Mask.tif' 
    img_root = '/disk2/Train_Tumor/'+img_name + '.tif'
    
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(xml_root)
    xml_repository.load()

    mr_image = reader.open(img_root)
    mr_image_mask = reader.open(mask_root)
    level = 0
    w, h = mr_image.getLevelDimensions(level)
    ds = mr_image.getLevelDownsample(level)
    
    image_patch = mr_image.getUCharPatch(0, 0, w, h, level)
    image_patch_mask = mr_image_mask.getUCharPatch(0, 0, w, h, level)
    
    anotation_list = [[n.getImageBoundingBox()[0].getX(), 
        n.getImageBoundingBox()[0].getY(),
        n.getImageBoundingBox()[1].getX(),
        n.getImageBoundingBox()[1].getY()] for n in annotation_list.getGroups()]
    
    for i in anotation_list:
        x_max,x_min ,y_max ,y_min  = i 

    true_label = image_patch_mask[int(x_min/ds):int(y_min/ds), int(x_max/ds):int(y_max/ds),0]
    true_img = image_patch[int(x_min/ds):int(y_min/ds), int(x_max/ds):int(y_max/ds),:]
    
    label_result = Image.fromarray(true_label)
    img_result = Image.fromarray(true_img)
    