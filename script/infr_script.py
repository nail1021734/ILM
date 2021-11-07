from utils.inference import inference

if __name__ == '__main__':
    sen = inference(
        ckpt_path='checkpoint/MLM_exp10/checkpoint-3200000.pt',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        prompt='[ARTICLE]<per0>育有<num>女<num>子,除了大女兒<per1>,小兒子<per2>也有在娛樂圈活動,[MASK_S],他哭笑不得地表示兒子有次無心的舉動,[MASK_S],「是不是幫我掛號讓我去問個醫生或怎麼樣?」後來才知道是烏龍一場。 <per0>最近在<unk>透露,<per2>曾讓房間內的電扇持續吹一整年,[MASK_S],一度想掛號就醫,一出門錄影卻又突然「好轉」,直到有次靠近兒子的房間時,耳鳴突然變得超大聲,他才終於抓到兇手! 往鋼琴底下一看,<per0>發現那裡有台電風扇,「連續吹風<num>年啊!」<per1>在旁邊有感而發地說:「這種男生要嘛只會關燈,[MASK_S],他沒辦法同時把這兩個地方都關了再出門!」 對此,<per2>事後解釋是因為擔心鋼琴會受潮,<per0>哭笑不得地說:「我終於在<per2>從<loc0>回<loc1>的時候,治好了我的耳鳴!恁爸差點就把他...!」對兒子是好氣又好笑。[SEP]',
        p=0.9
    )
    print(format(sen))
