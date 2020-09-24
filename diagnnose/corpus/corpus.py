from typing import List, Optional, Tuple, Union

from torchtext.data import Dataset, Example, Field, Pipeline, RawField
from transformers import PreTrainedTokenizer


class Corpus(Dataset):
    def __init__(
        self,
        examples: List[Example],
        fields: List[Tuple[str, Field]],
        create_pos_tags: bool = False,
        sen_column: str = "sen",
        labels_column: str = "labels",
    ) -> None:
        super().__init__(examples, fields)

        self.sen_column = sen_column
        self.labels_column = labels_column

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        if hasattr(self.fields[sen_column], "vocab"):
            self.tokenizer = self.fields[sen_column].vocab

        self.attach_sen_ids()

        # TODO: Fix when refactoring classifier module
        # if any(field_name == labels_column for field_name, _ in self.fields):
        #     self.fields[labels_column].build_vocab(self)

        if create_pos_tags:
            self.create_pos_tags()

    @classmethod
    def create(
        cls,
        path: str,
        header: Optional[List[str]] = None,
        header_from_first_line: bool = False,
        to_lower: bool = False,
        sen_column: str = "sen",
        labels_column: str = "labels",
        sep: str = "\t",
        tokenize_columns: Optional[List[str]] = None,
        convert_numerical: bool = False,
        create_pos_tags: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> "Corpus":
        raw_corpus = cls.create_raw_corpus(
            path, header_from_first_line=header_from_first_line, sep=sep
        )

        header = cls.create_header(
            header=header,
            header_from_first_line=header_from_first_line,
            corpus_path=path,
            sen_column=sen_column,
            labels_column=labels_column,
            sep=sep,
        )

        fields = cls.create_fields(
            header,
            to_lower=to_lower,
            sen_column=sen_column,
            labels_column=labels_column,
            tokenize_columns=tokenize_columns,
            convert_numerical=convert_numerical,
            tokenizer=tokenizer,
        )

        examples = cls.create_examples(raw_corpus, fields)

        return cls(
            examples,
            fields,
            create_pos_tags=create_pos_tags,
            sen_column=sen_column,
            labels_column=labels_column,
        )

    @staticmethod
    def create_raw_corpus(
        path: str, header_from_first_line: bool = False, sep: str = "\t"
    ) -> List[List[str]]:
        with open(path, encoding="utf8") as f:
            if header_from_first_line:
                next(f)
            raw_corpus = [line.strip().split(sep) for line in f]

        return raw_corpus

    @staticmethod
    def create_header(
        header: Optional[List[str]] = None,
        header_from_first_line: bool = False,
        corpus_path: Optional[str] = None,
        sen_column: str = "sen",
        labels_column: str = "labels",
        sep: str = "\t",
    ) -> List[str]:
        if header is None:
            if header_from_first_line:
                with open(corpus_path) as f:
                    header = next(f).strip().split(sep)
            elif corpus_path is not None:
                # Infer header from file structure
                with open(corpus_path) as f:
                    first_line = next(f).strip().split(sep)
                if len(first_line) == 2:
                    header = [sen_column, labels_column]
                else:
                    header = [sen_column]
            else:
                header = [sen_column]

        assert sen_column in header, "`sen` should be part of corpus_header!"

        return header

    @staticmethod
    def create_fields(
        header: List[str],
        to_lower: bool = False,
        sen_column: str = "sen",
        labels_column: str = "labels",
        tokenize_columns: Optional[List[str]] = None,
        convert_numerical: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> List[Tuple[str, Field]]:
        tokenize_columns = tokenize_columns or [sen_column]

        pipeline = None
        if convert_numerical:

            def preprocess_field(s: Union[str, int]) -> Union[str, int]:
                return int(s) if (isinstance(s, str) and s.isdigit()) else s

            pipeline = Pipeline(convert_token=preprocess_field)

        fields = []

        for column in header:
            if column in tokenize_columns:
                field = Field(batch_first=True, include_lengths=True, lower=to_lower)
                if tokenizer is not None:
                    attach_tokenizer(field, tokenizer)
            # TODO: fix when refactoring classifier module
            # elif column == labels_column:
            #     field = Field(
            #         pad_token=None,
            #         unk_token=None,
            #         is_target=True,
            #         preprocessing=pipeline,
            #     )
            else:
                field = RawField(preprocessing=pipeline)
                field.is_target = False

            fields.append((column, field))

        return fields

    @staticmethod
    def create_examples(
        raw_corpus: List[List[str]], fields: List[Tuple[str, Field]]
    ) -> List[Example]:
        examples = [Example.fromlist(line, fields) for line in raw_corpus]

        return examples

    def attach_sen_ids(self):
        self.fields["sen_idx"] = RawField()
        self.fields["sen_idx"].is_target = False

        for sen_idx, item in enumerate(self.examples):
            setattr(item, "sen_idx", sen_idx)

    def create_pos_tags(self):
        import nltk

        nltk.download("averaged_perceptron_tagger")

        self.fields["pos_tags"] = RawField()
        self.fields["pos_tags"].is_target = False

        print("Tagging corpus...")
        for item in self.examples:
            setattr(item, "pos_tags", [t[1] for t in nltk.pos_tag(item.sen)])


def attach_tokenizer(field: Field, tokenizer: PreTrainedTokenizer) -> None:
    """Creates a tokenizer that is attached to a Corpus Field.

    Parameters
    ----------
    field : Field
        Field to which the vocabulary will be attached
    tokenizer : PreTrainedTokenizer
        Tokenizer that will convert tokens to their index.
    """

    def preprocess(value: Union[str, List[str]]) -> List[str]:
        """We only perform the splitting as a preprocessing step.

        This allows us to still have access to the original tokens,
        including those that will be mapped to <unk> later.
        """
        if isinstance(value, list):
            value = " ".join(value)

        return [tokenizer.convert_ids_to_tokens(t) for t in tokenizer.encode(value)]

    field.preprocessing = preprocess
    field.pad_token = tokenizer.pad_token
    field.vocab = tokenizer
    field.vocab.stoi = tokenizer.vocab
