from utils.inference import inference, format

if __name__ == '__main__':
    sen = inference(
        ckpt_path='checkpoint/MLM_exp10/checkpoint-4086331.pt',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        prompt='[ARTICLE]前<loc0>市長<per0>、<org0>立委<per1><num>日合體,首度揭露「<per2>的秘密檔案」。<per1>首先自爆「最長部位」,[MASK_S],面對超勁爆的私人問題兩人絲毫不害臊,最後更直接示範「最愛姿勢」,引人遐想。 <per0>、<per1><num>日在臉書發布影片,[MASK_S],被問及「覺得自己哪理最長?」時,<per0>回答「腿」,<per1>則表示自己「氣最長」,[MASK_S],憋氣可以憋很久,另外就是「堅持」。 [MASK_S],<per0>爆料自己「全身都很敏感」,[MASK_S],所以皮膚是最敏感的地方。一聽到<per0>的回答,[MASK_S],因為「很喜歡聽老婆甜言蜜語」、「不是觸覺的那種,你們想歪了」<per0>也自嘲,「我太太說我耳朵很不敏感,因為她講什麼我常常都聽不到。」 [MASK_S],兩人直接示範「最喜歡的姿勢」,可見<per0>右手托著下巴<per1>則是雙手往前伸、比出「雙手緊握圈」的姿勢,原來是最喜歡跟選民「握手」。[SEP]',
        k=40,
        strategy='top-k'
    )
    print(format(sen))