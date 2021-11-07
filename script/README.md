# Script

此資料夾保存各功能的主程式，使用方法如下

## 建立tokenizer

使用huggingface的`tokenizer`提供的BPE方法訓練
```python
python -m script.build_tokenizer
```

## 建立ILM的資料

ILM的資料為將文章中的句子,token...使用特殊的MASK遮蓋過的資料，執行`create_mask_data.py`建立資料

```python
python -m script.create_mask_data
```

## 訓練模型

```python
python -m script.train_script
```

## Inference

```python
python -m script.infr_script
```

## 建立Machine News dataset

建立用來訓練disciminator的資料

```
python -m script.create_MN_data
```
