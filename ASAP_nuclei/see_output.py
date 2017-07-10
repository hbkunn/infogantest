import sys
sys.path.append('/opt/ASAP/bin')
import multiresolutionimageinterface as mir
reader = mir.MultiResolutionImageReader()
mr_image = reader.open('/disk2/Train_Tumor/Tumor_001.tif')
print(mr_image)
import wholeslidefilters as wsf
ndwsf = wsf.NucleiDetectionWholeSlideFilter()
ndwsf.setInput(mr_image)
ndwsf.setProcessedLevel(3)
ndwsf.setAlpha(2)
ndwsf.setBeta(0.01)
ndwsf.setThreshold(0.01)
ndwsf.setMaximumRadius(5.00)
ndwsf.setMinimumRadius(1.50)
ndwsf.setRadiusStep(1)
ndwsf.setOutput('/disk1/cell_work/ASAP_nuclei/nuclei.xml')
ndwsf.process()