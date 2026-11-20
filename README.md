# 中文评论情感分类

输入一条短评论，预测正面/负面。使用字符级 TF-IDF 与逻辑回归，提供多数类基线、独立测试集、混淆矩阵及逐条预测。适合展示一个完整的小型 NLP 分类流程。

## 运行

Python 3.11；在本项目目录执行（依赖清单随项目提供）：

```bash
python -m pip install -r requirements.txt
python main.py
python main.py --text "物流有点慢，但是东西很好用"
python main.py --output-dir outputs/demo
```

## 数据与评估

- `data/train.csv` 是 72 条自编中文教学评论，`data/test.csv` 是 24 条另写措辞的测试评论，两类均衡。`1` 表示正面，`0` 表示负面。没有抓取用户数据。
- 测试文件固定隔离；检查训练/测试完全相同的文本，词表和 IDF 只在训练集拟合。没有按模板批量替换商品名称扩增数据。
- 超参数事先固定，测试集不参与调参。比较 TF-IDF + Logistic Regression 与“总猜训练集多数类”的基线，报告准确率、正类 F1、Macro-F1。
- 数据规模很小且由同一作者设计，不能证明真实电商数据上的效果；否定、反讽、混合评价仍容易出错。输出概率未经校准。

## 输出与代码

`main.py` 包含载入、训练、评估、可视化和自定义预测，默认输出到脚本旁的 `outputs/`：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 本次实跑指标、基线、数据说明 |
| `predictions.csv` | 每条测试评论的标签与概率，可用于错误分析 |
| `confusion_matrix.png` | 混淆矩阵 |
| `demo.json` | `--text` 的预测 |

修改 CSV 后重新运行即可。增加数据时应保留独立测试来源；不要反复查看测试错误后把答案相似句补入训练集再宣称泛化提升。

## 面试可以讲什么

中文为什么可以用字符 n-gram、为什么把向量化器放在 Pipeline 中、类别不平衡时 Accuracy 与 Macro-F1 的区别，以及怎样用混淆矩阵找出模型误判的否定句。可进一步加入真实授权语料和独立验证集后再调参。

参考：[scikit-learn 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)。代码与自编数据采用本目录 MIT 许可。

## 实际示例结果

见[本次实际运行结果](RESULTS.md)，包含生成的指标和结果图。重新运行 `python main.py` 可以复现流程。
