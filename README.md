# Awesome Resources for NLP/NLG in Swedish [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

For those of you who are not used to long markdown files, GitHub gracefully generates a table of contents for you! See more info on how to find it [here](https://github.blog/changelog/2021-04-13-table-of-contents-support-in-markdown-files/).

## Corpora
Corpus = a collection of raw unlabeled texts

### Monolingual

#### Free and Downloadable
* [Språkbanken Text](https://spraakbanken.gu.se/en/resources/corpus) -- this is a hub page for many Swedish corpora maintained by the Språkbanken Text, monolingual corpora come from newspapers, blog posts, literature of different years (some from as early as the 18th century). **Note that many of these corpora contain scrambled sentences**.
* [CC-100](http://data.statmt.org/cc-100/) -- documents extracted from [Common Crawl](https://commoncrawl.org/), automatically classified and filtered. Swedish part is 21 GB of raw text.
* [mC4](https://github.com/allenai/allennlp/discussions/5056) -- a colossal, cleaned version of Common Crawl's web crawl corpus (C4), Swedish part contains about 65GB of raw text
* [SOU corpus](https://github.com/UppsalaNLP/SOU-corpus) -- cleaned and further processed versions of Swedish Government Official Reports (Statens offentliga utredningar, SOU), covers the reports between 1994 and 2020
* [SweDraCor](https://github.com/dracor-org/swedracor) -- corpus of 68 TEI-encoded Swedish language plays taken from [eDrama](https://litteraturbanken.se/dramawebben/sida/edrama) project
* [Swedish Poetry](https://github.com/aischeveva/swedish_poetry) -- poetry corpus
* [LBPF](https://github.com/mdahllof/lbpf) -- Swedish prose fiction with modern spelling from Litteraturbanken
* [SBS](https://www.ling.su.se/english/nlp/corpora-and-resources/sbs) -- a collection of sentences from Swedish blog posts from November 2010 until September 2012, **contains scrambled sentences** -- **NOTE: links seem to be broken as of 2022-05-25**
* [Project Runeberg](http://runeberg.org/katalog.html) -- copyright-free Swedish literature

#### Free and Available by Request
* [OSCAR](https://oscar-corpus.com/) — **scrambled** sentences extracted from [Common Crawl](https://commoncrawl.org/) and classified with a language detection model. It's Swedish portion comprises 48GB of raw text with roughly 7.5M documents and 5B words
* [Polyglot's processed Swedish Wikipedia](https://drive.google.com/file/d/0B5lWReQPSvmGNVZab2R6U3hqcU0/edit?usp=sharing)

### Parallel
* [OPUS](https://opus.nlpl.eu/) -- The Open Parallel Corpus, a hub for parallel datasets for many pairs of languages, including to/from Swedish.
* [Språkbanken Text](https://spraakbanken.gu.se/en/resources/corpus) -- this is a hub page for many Swedish corpora maintained by the Språkbanken Text, the available parallel corpora are EuroParl (Swedish part of European Parliament Proceedings Parallel Corpus) and ASPAC (Swedish part of The Amsterdam Slavic Parallel Aligned Corpus). **Note that both corpora contain scrambled sentences**.
* [SMULTRON](https://www.ling.su.se/english/nlp/corpora-and-resources/smultron) -- a parallel treebank that contains around 1000 sentences in English, German and Swedish

## Datasets
Dataset = a collection of labeled texts

### Monolingual

#### Free and Downloadable

##### Swedish-first
* Swedish Universal Dependencies treebanks -- can be used to train PoS-taggers, lemmatizers and dependency parsers
	- [Talbanken](https://github.com/UniversalDependencies/UD_Swedish-Talbanken/tree/master): 96K tokens, 6K sentences
	- [LinES](https://github.com/UniversalDependencies/UD_Swedish-LinES/tree/master): 90K tokens, 5K sentences
	- [PUD](https://github.com/UniversalDependencies/UD_Swedish-PUD/tree/master): 19K tokens, 1K sentences
* [SweQUAD-MC](https://github.com/dkalpakchi/SweQUAD-MC) -- a multiple choice question answering dataset
* [Swedish-sentiment](https://github.com/timpal0l/swedish-sentiment) -- a sentiment analysis dataset of 10000 texts with roughly 50/50 split between positive and negative sentiments
* [Swedish-Causality-Datasets](https://github.com/UppsalaNLP/Swedish-Causality-Datasets) -- namely causality recognition and causality ranking dataset, taking texts from the official reports of Swedish Government
* [Swedish-MWE-dataset](https://github.com/MurathanKurfali/swedish-mwe-dataset) -- a multiword expression dataset, containing 96 Swedish expressions annotated for their degrees of compositionality
* Swedish-NER
	- [by Andreas Klintberg](https://github.com/klintan/swedish-ner-corpus) -- semi-manually annotated Webbnyheter 2012 corpus from Språkbanken, 4 types of named entities: person, organization, location, miscellaneous.
	- [by Robert Lorentz](https://github.com/robban112/Swedish-NER-corpus)
	- [The Written Works Corpus](https://github.com/gilleti/the-written-work-corpus) -- named entities for written works: ART, BOOK, GAME, MOVIE, MUSIC, PLAY, RADIO and TV. A bit more detailed description about the corpus is [here](https://www.hillevihagglof.se/2018/09/08/the-written-work-corpus/)
* [SIC](https://www.ling.su.se/english/nlp/corpora-and-resources/sic) -- a corpus of Swedish Internet tags, manually annotated wth part of speech tags and named entities
* [SUSC](https://www.ling.su.se/english/nlp/corpora-and-resources/susc) -- a corpus of seven novels by August Strindberg annotated with part of speech tags with morphological analysis and lemmas
* [SNEC](https://www.ling.su.se/english/nlp/corpora-and-resources/snec) -- The Strindberg National Edition Corpus, both plain text version and linguistically annotated CoNLL-U version -- **NOTE: links seem to be broken as of 2022-05-25**
* [SuperLim](https://spraakbanken.gu.se/en/resources/superlim) -- a Swedish version of GLUE benchmark

##### Translated
* [OverLim](https://huggingface.co/datasets/KBLab/overlim) -- dataset contains some of the GLUE and SuperGLUE tasks automatically translated to Swedish, Danish, and Norwegian (bokmål), using the OpusMT models for MarianMT, **the translation quality was not manually checked**
* [XNLI](https://github.com/salesforce/xnli_extension) -- an autotranslated (Google Translate) natural language inference (NLI) dataset, **no info about human correction**
* [STS Benchmark](https://github.com/timpal0l/sts-benchmark-swedish) -- a semantic textual similarity (STS) dataset, automatically translated version of the original STS Benchmark for English using Google's NMT API **without human correction**
* [SwedSQuAD](https://github.com/Vottivott/swedsquad) -- a machine-translated version of SQuAD (Stanford Question Answering Dataset), **no info about human correction**

#### Free and Available By Request
* [SUC 2.0](https://spraakbanken.gu.se/index.php/en/resources/suc2) -- annotated with part-of-speech tags, morphological analysis and lemma (all that can be considered gold standard data), as well as some structural and functional information
* [SUC 3.0](https://spraakbanken.gu.se/index.php/en/resources/suc3) -- improved and extended SUC 2.0 

## Pre-trained resources

### Word embeddings
* Facebook's FastText vectors, 300-dimensional
	- trained on Common Crawl + Wikipedia: [vecs](https://fasttext.cc/docs/en/crawl-vectors.html)
	- trained on language-specific Wikipedia only: [vecs](https://fasttext.cc/docs/en/pretrained-vectors.html)
	- trained on Wikipedia with cross-lingual alignment: [vecs] (https://fasttext.cc/docs/en/aligned-vectors.html)
* [Diachronic embeddings](https://zenodo.org/record/4301658#.YoyHFjlByV5) from Språkbanken Text (based on word2vec and FastText)
* [NLPL repository](http://vectors.nlpl.eu/repository/) maintained by [Language Techonology Group](https://www.mn.uio.no/ifi/english/research/groups/ltg/) at the University of Oslo
	- Word2Vec, 100-dimensional: [vecs](http://vectors.nlpl.eu/repository/20/69.zip)
	- ELMo, 1024-dimensional: [vecs](http://vectors.nlpl.eu/repository/20/202.zip)
* [Swectors](https://www.ida.liu.se/divisions/hcs/nlplab/swectors/), 300-dimensional (the released vectors are Word2Vec)
* [Polyglot embeddings](https://sites.google.com/site/rmyeid/projects/polyglot): [vecs](http://bit.ly/19bTH2P)
* [Kyubyong Park's vectors](https://github.com/Kyubyong/wordvectors)
	- Word2Vec, 300-dimensional: [vecs](https://drive.google.com/open?id=0B0ZXk88koS2KNk1odTJtNkUxcEk)
	- FastText, 300-dimensional: [vecs](https://www.dropbox.com/s/7tbm0a0u31lvw25/sw.tar.gz?dl=0)
* [Flair embeddings](https://github.com/flairNLP/flair-lms), 2048-dimensional, can be used only within [flair](https://github.com/flairNLP/flair) package from Zalando Research

### Swedish-specific Transformer models
The code for calculating the number of parameters (comes from [this thread](https://github.com/huggingface/transformers/issues/1479)):
- PyTorch: `sum(p.numel() for p in model.parameters() if p.requires_grad)`
- TensorFlow: `np.sum([np.prod(v.shape) for v in tf.trainable_variables])`

And now to the models themselves, where the code snippet above was used to estimate the number of parameters.
* [BERT* models](https://github.com/Kungbib/swedish-bert-models) from The National Library of Sweden/KBLab
	- bert-base-swedish-cased: 12 layers, 768 hidden size, 12 heads, ~124M parameters
	- albert-base-swedish-cased-alpha: 12 layers, 768 hidden size, 12 heads, ~14M parameters
	- electra-small-swedish-cased
		- generator: 12 layers, 256 hidden size, 4 heads, ~16M parameters
		- discriminator: 12 layers, 256 hidden size, 4 heads, ~16M parameters
* [BERT models](https://github.com/af-ai-center/SweBERT) from Arbetsförmedlingen (The Swedish Public Employment Service) 
	- bert-base-swedish-uncased: 12 layers, 768 hidden size, 12 heads, ~110M parameters
	- bert-large-swedish-uncased: 24 layers, 1024 hidden size, 16 heads, ~335M parameters
* RoBERTa models
	- trained on Swedish Wikipedia and OSCAR: [model on HF Hub](https://huggingface.co/flax-community/swe-roberta-wiki-oscar)
	- trained on mC4: [model on HF Hub](https://huggingface.co/birgermoell/roberta-swedish-scandi)
	- seems to be trained on OSCAR?: [model on HF Hub](https://huggingface.co/birgermoell/roberta-swedish)
* GPT-2 models
	- trained on the Wiki40B and OSCAR: [model on HF Hub](https://huggingface.co/birgermoell/swedish-gpt)
	- trained on the Wiki40B only: [model on HF Hub](https://huggingface.co/flax-community/swe-gpt-wiki)
* GPT-SW3 model (3.5B parameters): [model on HF Hub](https://huggingface.co/AI-Nordics/gpt-sw3) -- **NOTE: The repository is empty as of 2022-08-23**
* T5 models
	- trained on OSCAR: [model on HF Hub](https://huggingface.co/birgermoell/t5-base-swedish)

### Nordic Transformer models
* GPT-2 models:
	- trained on Wiki40B: [model on HF Hub](https://huggingface.co/flax-community/nordic-gpt-wiki)

### Multilingual Transformer models
* [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) -- multilingual BERT by Google Research
* [mBART50](https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual#mbart50-models) -- multilingual BART by FAIR
* [mT5](https://github.com/google-research/multilingual-t5) -- multilingual T5 by Google Research

### Dependency parsing models
* [Stanza's models](https://stanfordnlp.github.io/stanza/available_models.html) -- trained on UD treebanks: one on Talbanken and another on LinES
* [MaltParser](https://www.maltparser.org/mco/mco.html)

### Part of speech taggers
* [Stagger](https://www.ling.su.se/english/nlp/tools/stagger/stagger-the-stockholm-tagger-1.98986)

### Machine Translation models to/from Swedish
* OPUS-MT models: [models on HF Hub](https://huggingface.co/models?language=sv&pipeline_tag=translation&sort=modified)

## Tools
* [Granska](http://skrutten.csc.kth.se/) -- software for grammar control
* [Stava](https://www.csc.kth.se/~viggo/stava/) -- software for spell checking

## Other resources
* Wordlists -- [here](https://github.com/almgru/svenska-ord.txt), [here](https://github.com/martinlindhe/wordlist_swedish) or [here](https://github.com/Maistho/wordlists)
