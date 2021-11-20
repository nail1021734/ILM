from utils.inference import format_article, inference

if __name__ == '__main__':
    sen = inference(
        ckpt_path='checkpoint/MLM_exp11/checkpoint-3360000.pt',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        prompts=['[ARTICLE]許多人會抱怨,為什麼自己的退休金那麼少,只有別人的一半而已?在<unk>節目中,來賓們提出五點一般人在投資上常有的迷思,如果想存滿退休金,很重要的就是打破這迷思! <num>.定存比股票安全? 闕又上說:「定存的優點是短期很安全,[MASK_S],但長期是安全的」。以<num>法則來看,定存報酬率<num>,翻倍時間要<num>年,但台股<num>,報酬率<num>,[MASK_S]。[MASK_S],長期下來實在是差之千里。 <num>.高收益代表好投資? 闕又上說:「高收益債原則上不適合大多數人,因為它是股票跟公債的混合體,[MASK_S],收益債體質也不高,所以要投資的話,不如就美國標普<num>和美國政府公債的混搭,這樣獲利可以達到<num>倍」。 <num>.避開波動就是安全投資? [MASK_S],波動就是機會。闕又上認為,波動不一定是風險,反而是中性的,股市有波動是正常的,[MASK_S]。 <num>.避免套牢,停損可增加投資績效? 「不敗教主」陳重銘說:「會想停利或停損,代表你不了解那間公司,[MASK_S],所以建議投資前,一定要對公司有一定了解」。 <num>.高額報酬才能致富? 陳重銘表示,高額報酬不一定能致富,[MASK_S],所以建議還是慢慢走,靠著複利績效累積資產。[SEP]'],
        k=40,
    )
    print(format_article(sen[0]))
