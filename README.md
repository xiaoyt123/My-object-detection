一、数据集准备
1：从数据提供的XML文件中，提取出每张图片voc格式的xml文件，文件存放在Annotations，代码为DETRAC_xmlParser.py 
2：根据生成的XML文件，迁移图片到目标目录中，代码为voc_data_migrate.py
3：利用代码ImageSets_Convert.py, 产生trainval.txt,test.txt,train.txt,val.txt文件，文件存放在Main中

数据集的导入和预处理：datasets.py
训练：train.py
测试：test.py
