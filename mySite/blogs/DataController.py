import numpy as np
import cv2
import os
import Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator  


def PreprocessingAllImages(IMAGE_WIDTH = 2048, IMAGE_HEIGHT = 2048):
    pathFolder=[]   
    pathFolderGenerate=[]   
    destinationFolder=[] 
    labelName=[]
    data = []
    labels = []
    classification=[]
    pathFolder.append("DATASET/0.Normal/")
    pathFolder.append("DATASET/1.Tessellated fundus/")
    pathFolder.append("DATASET/2.Pathological myopia/")
    pathFolder.append("DATASET/3.CRVO/") 
    pathFolder.append("DATASET/4.Chorioretinal atrophy-coloboma/")
    pathFolder.append("DATASET/5.Vessel tortuosity/")
    pathFolder.append("DATASET/6.Massive hard exudates/")
    pathFolder.append("DATASET/7.Severe hypertensive retinopathy/")
    pathFolder.append("DATASET/8.Retinitis pigmentosa/") 
    pathFolder.append("DATASET/9.DR3/")
    pathFolder.append("DATASET/10.Optic atrophy/")
    pathFolder.append("DATASET/11.Blur fundus/")

    pathFolderGenerate.append("DATASET Generated/0.Normal/")
    pathFolderGenerate.append("DATASET Generated/1.Tessellated fundus/")
    pathFolderGenerate.append("DATASET Generated/2.Pathological myopia/")
    pathFolderGenerate.append("DATASET Generated/3.CRVO/") 
    pathFolderGenerate.append("DATASET Generated/4.Chorioretinal atrophy-coloboma/")
    pathFolderGenerate.append("DATASET Generated/5.Vessel tortuosity/")
    pathFolderGenerate.append("DATASET Generated/6.Massive hard exudates/")
    pathFolderGenerate.append("DATASET Generated/7.Severe hypertensive retinopathy/")
    pathFolderGenerate.append("DATASET Generated/8.Retinitis pigmentosa/") 
    pathFolderGenerate.append("DATASET Generated/9.DR3/")
    pathFolderGenerate.append("DATASET Generated/10.Optic atrophy/")
    pathFolderGenerate.append("DATASET Generated/11.Blur fundus/")
    
    destinationFolder.append("DATASET Preprocess/0.Normal/")
    labelName.append("Normal") 
    destinationFolder.append("DATASET Preprocess/1.Tessellated fundus/")
    labelName.append("Tessellated fundus")  
    destinationFolder.append("DATASET Preprocess/2.Pathological myopia/")
    labelName.append("Pathological myopia")
    destinationFolder.append("DATASET Preprocess/3.CRVO/")
    labelName.append("CRVO")
    destinationFolder.append("DATASET Preprocess/4.Chorioretinal atrophy-coloboma/")
    labelName.append("Chorioretinal atrophy-coloboma")
    destinationFolder.append("DATASET Preprocess/5.Vessel tortuosity/")
    labelName.append("Vessel tortuosity") 
    destinationFolder.append("DATASET Preprocess/6.Massive hard exudates/")
    labelName.append("Massive hard exudates")
    destinationFolder.append("DATASET Preprocess/7.Severe hypertensive retinopathy/")
    labelName.append("Severe hypertensive retinopathy")
    destinationFolder.append("DATASET Preprocess/8.Retinitis pigmentosa/")
    labelName.append("Retinitis pigmentosa") 
    destinationFolder.append("DATASET Preprocess/9.3.DR3/")
    labelName.append("DR3")    
    destinationFolder.append("DATASET Preprocess/10.Optic atrophy/")
    labelName.append("Optic atrophy")
    destinationFolder.append("DATASET Preprocess/11.Blur fundus/")
    labelName.append("Blur fundus")  
    
    NumberGenerate = 6
    for GeneFile in pathFolderGenerate:
        if not os.path.exists(GeneFile):
            os.mkdir(GeneFile)
            File = pathFolder[pathFolderGenerate.index(GeneFile)]
            imagesFile=[x for x in os.listdir(File) if os.path.isfile(os.path.join(File,x))]
            for image in imagesFile:
                imageNname = os.path.splitext(image)[0]
                imagePath = File+'/'+image	
                im=image = cv2.imread(imagePath) 
                gen = ImageDataGenerator(rotation_range=NumberGenerate, width_shift_range=1, 
                                         height_shift_range=1, shear_range=1.5, zoom_range=0.1, 
                                         channel_shift_range=10., horizontal_flip=True)
                aug_iter = gen.flow(np.array(im).reshape(1,np.array(im).shape[0] , np.array(im).shape[1],3))
                aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(NumberGenerate)]
                for i in range(NumberGenerate):
                    cv2.imwrite(GeneFile+imageNname+"BloodVessels"+str(i+1)+".jpg",aug_images[i])
    for destFile in destinationFolder:
        if not os.path.exists(destFile):
            os.mkdir(destFile)
            File = pathFolder[destinationFolder.index(destFile)]
            GeneFile = pathFolderGenerate[destinationFolder.index(destFile)]
            imagesFile=[x for x in os.listdir(File) if os.path.isfile(os.path.join(File,x))]
            imagesFileGene=[x for x in os.listdir(GeneFile) if os.path.isfile(os.path.join(GeneFile,x))]
          
            for image in imagesFile:
                imageNname = os.path.splitext(image)[0]
                imagePath = File+'/'+image	
                img = cv2.imread(imagePath) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                img_t= Preprocessing.GaussianBlurfilter(img,100)
                img_t = cv2.resize(img_t, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
                BloodVessels = Preprocessing.VesselDetection(img_t,img_t)
                cv2.imwrite(destFile+imageNname+"BloodVessels.jpg",BloodVessels)
            
            for image in imagesFileGene:
                imageNname = os.path.splitext(image)[0]
                imagePath = GeneFile+'/'+image	
                img = cv2.imread(imagePath) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                img_t= Preprocessing.GaussianBlurfilter(img,100)
                img_t = cv2.resize(img_t, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
                BloodVessels = Preprocessing.VesselDetection(img_t,img_t)
                cv2.imwrite(destFile+imageNname+"BloodVessels.jpg",BloodVessels)
    imagen=[]          
    for destFile in destinationFolder:
       classi = np.zeros(len(labelName) , dtype="uint8")
       imagesFile=[x for x in os.listdir(destFile) if os.path.isfile(os.path.join(destFile,x))]
       for image in imagesFile:
           imageNname = os.path.splitext(image)[0]
           imagePath = destFile+'/'+image	
           image = cv2.imread(imagePath)
           image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
           data.append(image)
           labels.append(labelName[destinationFolder.index(destFile)])
           imagen.append(imageNname)
           classi[destinationFolder.index(destFile)]=1
           classification.append(classi) 
    return data, labelName,classification  ,imagen
