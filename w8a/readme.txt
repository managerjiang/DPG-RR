文件说明：
1.数据集
w8a.mat 是训练集数据（生成A矩阵[49749x123]）
L.mat 是训练集标签（生成A矩阵[1x49749]）
w8a_test.mat 是测试集数据（生成A矩阵[49749x123]）
L_test.mat 是测试集标签（生成A矩阵[1x49749]）
w8a_smote.mat是smote之后的训练集数据（生成A矩阵[95598x123]）
L_smote.mat是smote之后的训练集标签（生成A矩阵[1x95598]）
data/C_meth1_smote_sw_800.mat 保存的是切换拓扑邻接矩阵
3.功能程序
drawthem.m绘图程序
smote.m：另一种新的数据处理方法，对原始数据正负样本差距过大进行纠正。具体原理：smote法，在7800个正样本里面，依次for循环，每次循环找距离该点最近的m个点，随机选其中一个连线，再在连线上随机找1个点作为插值点，重复k次。（由于负样本大约是49749-1479个，因此k取32，m取10，拓充后的数据约为95598个，正负样本比例约为0.97：1）