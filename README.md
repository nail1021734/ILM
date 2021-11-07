# ILM
訓練方法參考自[Enabling Language Models to Fill in the Blanks](https://aclanthology.org/2020.acl-main.225/)

## 程式架構

分成3個部份
1. [dataset_script](./dataset_script/README.md): 用來讀取存放在`dataset`資料夾下的資料(使用huggingface提供的`load_dataset`方法讀取)
2. [script](./script/README.md): 存放執行training, evaluation, inference或創建資料時的主程式
3. [utils](./utils/README.md): 存放建立config方法, 讀取資料, 訓練模型時的副程式
