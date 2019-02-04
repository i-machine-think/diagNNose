# Example directory

In this directory, you can download several examples that illustrate how to use `rnnalyse`.
Currently, all the examples are based on the pretrained model provided by Gulordava et al (2018).
To download the files required for the examples, first run the file `setup.sh` in this folder:

```
sh setup.sh
```

# Extract

To extract activations from the downloaded model, you can use the script `extract.py`:

```
python3 extract.py -c extract.json
```

Explain what the input and output of this look like and what will be written to where.

# Diagnose

Explain how this file works/what it does.

## References

K. Gulordava, P. Bojanowski, E. Grave, T. Linzen, M. Baroni. 2018. [https://arxiv.org/abs/1803.11138](Colorless green recurrent networks dream hierarchically). Proceedings of NAACL.
