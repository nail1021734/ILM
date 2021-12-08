from utils.inference import format_article, inference

if __name__ == '__main__':
    sen = inference(
        ckpt_path='checkpoint/MLM_exp12/checkpoint-3120000.pt',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        prompts=['[ARTICLE]史上首位身價突破美金<num>億元的嘻哈歌手<per0>,[MASK_S],特地準備了價值美金<num>萬元的勞力士<en>系列手錶以及美金<num>元的香檳。[MASK_S],一系列的誇張「炫富」表現,[MASK_S]:「<en>-處於另一個級別。」 [MASK_S],<num>人皆收到<per0>的<en>慈善基金會晚宴邀請,[MASK_S],<per1>表示:「[MASK_S],[MASK_S]」,而史威茲畢茲則讚嘆<per0>的級別就是與他人不同,並且除了手錶以外,[MASK_S]。 由<per0>所舉辦的<en>慈善基金會晚宴,將於<num>/<num>與<num>/<num>在塞米諾爾硬石酒店與賭場舉辦,而<en>邀請函正是勞力士的<en>系列手錶。[SEP]'],
        k=40,
    )
    print(format_article(sen[0]))
