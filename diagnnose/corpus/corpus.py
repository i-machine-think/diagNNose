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
        labels_column: Optional[str] = None,
    ) -> None:
        super().__init__(examples, fields)

        self.sen_column = sen_column
        self.labels_column = labels_column

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        if hasattr(self.fields[sen_column], "vocab"):
            self.tokenizer = self.fields[sen_column].vocab

        self._attach_sen_ids()

        if create_pos_tags:
            self._create_pos_tags()

    @classmethod
    def create(
        cls,
        path: str,
        header: Optional[List[str]] = None,
        header_from_first_line: bool = False,
        to_lower: bool = False,
        sen_column: str = "sen",
        labels_column: Optional[str] = None,
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
            sep=sep,
        )

        fields = cls.create_fields(
            header,
            to_lower=to_lower,
            sen_column=sen_column,
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
        sep: str = "\t",
    ) -> List[str]:
        if header is None:
            if header_from_first_line:
                with open(corpus_path) as f:
                    header = next(f).strip("\n").split(sep)
            else:
                header = [sen_column]

        assert sen_column in header, f"{sen_column} should be part of corpus_header!"

        return header

    @staticmethod
    def create_fields(
        header: List[str],
        to_lower: bool = False,
        sen_column: str = "sen",
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

    def slice(self, sen_ids: List[int]) -> "Corpus":
        """Returns a new Corpus only containing examples from sen_ids.

        Parameters
        ----------
        sen_ids : List[int]
            List of sentence indices based on which the examples in the
            current Corpus will be filtered. These indices refer to the
            sen_idx in the original corpus; the newly sliced corpus
            will retain the original sen_idx of an Example item.

        Returns
        -------
        subcorpus : Corpus
            A new Corpus instance containing the filtered list of
            Examples.
        """
        examples = [ex for ex in self.examples if ex.sen_idx in sen_ids]

        subcorpus = Corpus(
            examples,
            self.fields,
            sen_column=self.sen_column,
            labels_column=self.labels_column,
        )

        return subcorpus

    def _attach_sen_ids(self):
        """ Adds a sentence index field to the Corpus. """
        self.fields["sen_idx"] = RawField()
        self.fields["sen_idx"].is_target = False

        for sen_idx, item in enumerate(self.examples):
            setattr(item, "sen_idx", sen_idx)

    def _create_pos_tags(self):
        """ Attaches nltk POS tags to the corpus for each sentence. """
        import nltk

        nltk.download("averaged_perceptron_tagger")

        self.fields["pos_tags"] = RawField()
        self.fields["pos_tags"].is_target = False

        print("Tagging corpus...")
        for item in self.examples:
            sen = getattr(item, self.sen_column)
            setattr(item, "pos_tags", [t[1] for t in nltk.pos_tag(sen)])


def attach_tokenizer(field: Field, tokenizer: PreTrainedTokenizer) -> None:
    """Creates a tokenizer that is attached to a Corpus Field.

    Parameters
    ----------
    field : Field
        Field to which the vocabulary will be attached
    tokenizer : PreTrainedTokenizer
        Tokenizer that will convert tokens to their index.
    """

    def preprocess(text: Union[str, List[str]]) -> List[str]:
        """We only perform the splitting as a preprocessing step.

        This allows us to still have access to the original tokens,
        including those that will be mapped to <unk> later.

        We cast the encoded text back to tokens for debugging purposes,
        making it easier to inspect an example at a later stage.
        """
        if isinstance(text, list):
            text = " ".join(text)

        return tokenizer.convert_ids_to_tokens(
            tokenizer.encode(text, add_special_tokens=True)
        )

    field.preprocessing = preprocess
    field.pad_token = tokenizer.pad_token
    field.vocab = tokenizer
    field.vocab.stoi = tokenizer.vocab
