from diagnnose.config import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor
from diagnnose.models import LanguageModel, import_model
from diagnnose.tokenizer.create import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    tokenizer = create_tokenizer(**config_dict["tokenizer"])
    corpus: Corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])
    model: LanguageModel = import_model(**config_dict["model"])

    # extractor = Extractor(model, corpus, **config_dict["extract"])
    # a_reader = extractor.extract()
    #
    # a_reader.cat_activations = True
    #
    # activations = print(a_reader[:10])

    import torch

    sen1 = "The hero forgot the tea.</s></s> A candidate leads a company."
    sen2 = "The tea was forgotten by the hero.</s></s> A candidate leads a company."

    tok1 = torch.tensor([tokenizer(sen1)["input_ids"]])
    tok2 = torch.tensor([tokenizer(sen2)["input_ids"]])

    start_idx1 = 9
    start_idx2 = 11

    print(
        [tokenizer.convert_ids_to_tokens(idx) for idx in tokenizer(sen1)["input_ids"]]
    )
    print(
        [tokenizer.convert_ids_to_tokens(idx) for idx in tokenizer(sen2)["input_ids"]]
    )

    with torch.no_grad():
        log1 = torch.log_softmax(
            model(input_ids=tok1, compute_out=True, mask_idx=tokenizer.mask_token_id),
            dim=-1,
        )[:, range(tok1.shape[1]), tok1].squeeze()
        log2 = torch.log_softmax(
            model(input_ids=tok2, compute_out=True, mask_idx=tokenizer.mask_token_id),
            dim=-1,
        )[:, range(tok2.shape[1]), tok2].squeeze()

    print(log1, log1[start_idx1:].sum())
    print(log2, log2[start_idx2:].sum())

# x_px,x_py,y_px,y_py,
# logp_x_px,logp_x_py,logp_y_px,logp_y_py,ppl_x_px,ppl_x_py,ppl_y_px,ppl_y_py
# <s>The woman judged the parent. A candidate leads a company.,
# <s>The parent was judged by the woman. A candidate leads a company.,
# <s>The woman judged the parent. A company is led by a candidate.,
# <s>The parent was judged by the woman. A company is led by a candidate.,
# -78.70220947265625,
# -76.26138305664062,
# -102.44937896728516,
# -107.46493530273438,
# 497343.03125,
# 331118.28125,
# 370727.59375,inf
