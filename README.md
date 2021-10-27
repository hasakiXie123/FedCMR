# FedCMR: Federated Cross-Modal Retrieval  

This repository is the official implementation of [FedCMR: Federated Cross-Modal Retrieval](https://dl.acm.org/doi/10.1145/3404835.3462989)

## 目录介绍

> FedCMR
>
> > main.py 主逻辑文件
> >
> > model.py 模型文件
> >
> > train.py 训练模型
> >
> > data 存放数据集
> >
> > data_loader.py 加载数据集
> >
> > utils.py 工具函数文件
> >
> > option.py 配置文件
> >
> > evaluate.py 计算mAP
> >
> > requirements.txt 依赖文件

## 数据处理

1. 如何处理？

   * 图片

     使用keras内置的VGG19 <sup>1</sup> 预训练模型学习图片特征，每张图片提取出一个[1,4096]大小的特征向量。

   * 文本

     使用sentence BERT-Large, Uncased<sup>2</sup> 学习文本特征，每个文本提取出一个[1,1024]大小的特征向量。

2. 选择的数据集

   ![datasets](https://i.loli.net/2021/07/11/IWSdYeHjLzqZoiM.png)

   + Pascal Sentence dataset<sup>3</sup> 
     - 特征向量文件: FedCMR/data/Pascal
   + Wikipedia dataset<sup>4</sup>
     - 特征向量文件: FedCMR/data/Wikipedia
   + MIR-Flickr25K dataset<sup>5</sup>
     - 特征向量文件: FedCMR/data/MIR-Flickr25K
   + MS-COCO dataset<sup>6</sup>
     - 特征向量文件: FedCMR/data/MS-COCO
   + 我们提供Pascal数据集的特征文件。

3. References

   [1] KarenSimonyanandAndrewZisserman.2014.Verydeepconvolutionalnetworks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014).

   [2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

   [3] Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. 2010. Collecting image annotations using amazon’s mechanical turk. In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and Language Data with Amazon’s Mechanical Turk. 139–147.

   [4] NikhilRasiwasia,JoseCostaPereira,EmanueleCoviello,GabrielDoyle,GertRG Lanckriet, Roger Levy, and Nuno Vasconcelos. 2010. A new approach to cross- modal multimedia retrieval. In Proceedings of the 18th ACM international confer- ence on Multimedia. 251–260.

   [5] Mark J Huiskes and Michael S Lew. 2008. The mir flickr retrieval evaluation. In Proceedings of the 1st ACM international conference on Multimedia information retrieval. 39–43.

   [6] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European conference on computer vision. Springer, 740–755.

## 第三方库

* 拉取代码并在FedCMR目录下运行 **pip3 install -r requirements.txt**
* 请特别注意在代码中，我们使用了[PySyft库的0.2.9版本](https://github.com/OpenMined/PySyft/tree/PySyft/syft_0.2.x)（该版本目前已经不再被官方支持），只有安装该版本的PySyft库才能成功运行代码。

## 有问题反馈

* 如有任何问题，欢迎反馈给我，可以用以下联系方式跟我交流
  * 邮箱：<qiujie_xie@126.com>


