各个文件说明：

celebA_data_preprocess.py: 人脸数据集预处理，将图片缩放到64*64的大小
dcgan_celeb.py: keras实现GAN生成人脸
degan_minst.py:keras实现GAN生成手写数字
rgan.py: keras实现RGAN生成正弦曲线
rgan_ab.py: keras实现RGAN生成异常曲线
utils.py：各个数据集加载工具库

运行方式：
Python rgan_ab.py -mode train —batch_size 64
mode包括：train和test
batchsize根据具体网络来定

Gan中出现的问题：
1.模式崩溃，生成器容易生成相似的样本
2.训练陷入局部最优点，很久才能出来，此时生成器和判别器的loss基本不变。
3.出现奇怪bug：手写数字训练时能够将真假样本组成一个batch进行训练，人脸和后续的不行，必须分两个batch训练。
