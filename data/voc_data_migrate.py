
# 然后根据生成的vocxml,迁移相应的图片到目标目录

import os
import random
import shutil
 
#xml路径的地址
XmlPath=r'My-object-detection/data/xml_test'
#原图片的地址
pictureBasePath=r"My-object-detection/data/Insight-MVT_Annotation_Train"
#保存图片的地址
saveBasePath=r"My-object-detection/data/image_test"
 
total_xml = os.listdir(XmlPath)
num=len(total_xml)
list=range(num)
if os.path.exists(saveBasePath)==False: #判断文件夹是否存在
     os.makedirs(saveBasePath)
 
 
for xml in total_xml:
    xml_temp=xml.split("__")
    folder=xml_temp[0]
    filename=xml_temp[1].split(".")[0]+".jpg"
    # print(folder)
    # print(filename)
    temp_pictureBasePath=os.path.join(pictureBasePath,folder)
    filePath=os.path.join(temp_pictureBasePath,filename)
    # print(filePath)
    newfile=xml.split(".")[0]+".jpg"
    newfile_path=os.path.join(saveBasePath,newfile)
    print(newfile_path)
    shutil.copyfile(filePath, newfile_path)
print("xml file total number",num)