# StackOverflow潜在专家预测 实验测试

## 环境配置
1. Ubuntu系统下，运行 **setup.sh** 下载数据，或使用数据生成部分自行生成数据。其他系统可根据requirements.txt安装依赖库，并生成或下载数据。
2. 解压下载的".7z"压缩包到"Data/"。 示例:`7za x Data/StackExpert.7z -oData`

## 运行

运行"main.py"，可选择以下几种功能：
1. 网格搜索法搜索参数。
   
   模型会随机生成符合条件的参数网格，并寻找最优参数。参数网格的限制可在"Data/params.json"中修改。实验使用了四种分类器（梯度提升树(GBDT)、极度随机森林、支持向量机、岭回归）和ADASYN过采样算法。
   (实验会持续循环进行，可使用<kbd>Ctrl</kbd>+<kbd>C</kbd> 退出)

2. 寻找最可能的潜在专家。

    模型会根据网格搜索的结果使用最优参数，预测出潜在专家概率最大的20名用户并显示他们的ID。

3. 显示最优参数。

    根据网格搜索的结果筛选出最优参数，并直接显示出来。

4. 模型性能分析。

    模型会根据网格搜索的结果使用最优参数，绘制准确率-召回率曲线，并输出特征重要性的数值。

在运行足够时长的功能[1]后，才能在功能[2-4]中获得较为准确的结果。


## 数据

Expert data:

[Google Drive](https://drive.google.com/open?id=1u1iTWKbG2v6TvxCRQHvgOnzBC0ib0N5K)

[Baidu Yun](https://pan.baidu.com/s/16xhoyJ_4FggdyvpvFXed1w) (密码:7jqb)