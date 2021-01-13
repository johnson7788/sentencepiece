# SentencePiece

[![Build Status](https://travis-ci.org/google/sentencepiece.svg?branch=master)](https://travis-ci.org/google/sentencepiece)
[![Build status](https://ci.appveyor.com/api/projects/status/vxoub3qx4fwpysyq?svg=true)](https://ci.appveyor.com/project/taku910/sentencepiece)
[![Coverage Status](https://coveralls.io/repos/github/google/sentencepiece/badge.svg?branch=master)](https://coveralls.io/github/google/sentencepiece?branch=master)
[![GitHub Issues](https://img.shields.io/github/issues/google/sentencepiece.svg)](https://github.com/google/sentencepiece/issues)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5851945fc54947fc9e964f78c3b6bdfa)](https://app.codacy.com/app/taku910/sentencepiece?utm_source=github.com&utm_medium=referral&utm_content=google/sentencepiece&utm_campaign=Badge_Grade_Dashboard)
[![PyPI version](https://badge.fury.io/py/sentencepiece.svg)](https://badge.fury.io/py/sentencepiece)
[![PyPi downloads](https://img.shields.io/pypi/dm/sentencepiece?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sentencepiece/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

SentencePiece是一种无监督的文本tokenizer和detokenizer工具，主要用于基于神经网络的文本生成系统，在神经模型训练之前预先确定了单词表量。 
SentencePiece实现了
**字词单元subword units** (e.g., **byte-pair-encoding (BPE)字节对编码** [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)]) 和
**unigram language model** [[Kudo.](https://arxiv.org/abs/1804.10959)])，一元语言模型。
SentencePiece从原始句子扩展直接训练。 SentencePiece使我们能够制作一个不依赖于特定于语言的预处理/后处理的纯粹的端到端系统。 

**This is not an official Google product.**

## 技术亮点
- **纯数据驱动**: SentencePiece从句子中训练tokenization和detokenization模型. Pre-tokenization ([Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)/[MeCab](http://taku910.github.io/mecab/)/[KyTea](http://www.phontron.com/kytea/)) 并非总是必需的 .
- **语言独立性**: SentencePiece将句子视为Unicode字符序列。 没有依赖于语言的逻辑。 即支持任何语种，包含英文，德语，中文，日语等
- **支持多种子词算法**: **BPE**  [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)] and **unigram language model** [[Kudo.](https://arxiv.org/abs/1804.10959)] are supported.
- **子词 regularization正则**: SentencePiece 实现子词采样通过 [subword regularization](https://arxiv.org/abs/1804.10959) and [BPE-dropout](https://arxiv.org/abs/1910.13267) 这有助于提高NMT模型的鲁棒性和准确性 .
- **快速和轻量**: Segmentation分段速度约为5万句/秒，内存占用约为6MB 
- **Self-contained**: 只要使用相同的模型文件，就可以获得相同的tokenizer/detokenizer。
- **直接单词表id生成**: SentencePiece管理单词表到ID的映射，并可以直接从原始句子生成单词表ID序列。
- **NFKC-based normalization**: SentencePiece 执行 NFKC-based 文本标准化 normalization.

对于不熟悉 SentencePiece as a software/algorithm的人, 可以阅读 [a gentle introduction here](https://medium.com/@jacky2wong/understanding-sentencepiece-under-standing-sentence-piece-ac8da59f6b08).


## 对比其它实现方法
|Feature|SentencePiece|[subword-nmt](https://github.com/rsennrich/subword-nmt)|[WordPiece](https://arxiv.org/pdf/1609.08144.pdf)|
|:---|:---:|:---:|:---:|
|Supported algorithm|BPE, unigram, char, word|BPE|BPE*|
|OSS?|Yes|Yes|Google internal|
|Subword regularization|[Yes](#subword-regularization)|No|No|
|Python Library (pip)|[Yes](python/README.md)|No|N/A|
|C++ Library|[Yes](doc/api.md)|No|N/A|
|Pre-segmentation required?|[No](#whitespace-is-treated-as-a-basic-symbol)|Yes|Yes|
|Customizable normalization (e.g., NFKC)|[Yes](doc/normalization.md)|No|N/A|
|Direct id generation|[Yes](#end-to-end-example)|No|N/A|

请注意，WordPiece中使用的BPE算法与原始BPE略有不同。 

## 总览
### 什么是 SentencePiece?
SentencePiece是对**sub-word units子词单元**的重新实现，是缓解神经机器翻译中开放单词表问题的有效方法。 
SentencePiece支持两种分段segmentation算法，即字节对编码(BPE)** [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)] 和unigram语言模型**unigram language model** [[Kudo.](https://arxiv.org/abs/1804.10959)]。 这是与其他实现的高级差异。

#### 唯一token的数量是预定的，需要首先确定单词表的数目，超参数
神经机器翻译模型通常以固定的单词表运行。 
与大多数无监督分词算法假设无限的单词表量不同，
SentencePiece训练分词模型最终单词表量是固定的，例如8k，16k或32k，例如BERT的中文单词表设置的是20k个

请注意，SentencePiece指定了用于训练的最终单词表量，这与使用合并操作数的[subword-nmt](https://github.com/rsennrich/subword-nmt)不同。 
合并操作的数量是特定于BPE的参数，不适用于其他分段算法，包括unigram，word and character。

#### 原始句子中的训练 
先前的子词实现假定输入句子已pre-tokenized,即已经进行了分词。 此约束是进行有效训练所需的约束，但由于我们必须提前运行与语言相关的tokenizer，因此预处理变得很复杂。
SentencePiece的实现足够快，可以从原始句子训练模型。 这对于训练单词之间没有显着空格的中文和日语的分词器和分词器很有用。 

#### 空格被视为基本符号
自然语言处理的第一步是文本tokenization。 
例如，标准的英语tokenizer将分割文本“Hello world.” 为以下三个token。

> [Hello] [World] [.]

一种观察是原始输入和tokenized序列是“不可逆转换的”。 
例如，在 “World” and “.”之间没有空格的信息被从tokenized序列中删除，
所以产生`Tokenize(“World.”) == Tokenize(“World .”)`

SentencePiece将输入文本视为Unicode字符序列。 空格也被视为普通符号。 
为了明确地将空白作为基本token来处理，SentencePiece首先使用元符号"▁" (U+2581)来转义空白，如下所示。 

> Hello▁World.
然后，将此文本分成小段，例如： 

> [Hello] [▁Wor] [ld] [.]

由于空格保留在分割后的文本中，因此我们可以毫无歧义地对文本进行detokenize。 

```
  detokenized = ''.join(pieces).replace('▁', ' ')
```

此特征使得无需依赖特定于语言的资源即可执行detokenizer。 

请注意，使用标准分词器分割句子时，我们无法应用相同的无损转换，因为它们将空格视为特殊符号。 Tokenized序列不会保留恢复原始句子所需的信息。

* (en) Hello world.   → [Hello] [World] [.]   \(A space between Hello and World\)
* (ja) こんにちは世界。  → [こんにちは] [世界] [。] \(No space between こんにちは and 世界\)

#### Subword regularization and BPE-dropout, 子词正则和BEP-dropout
子词正则化[[Kudo.](https://arxiv.org/abs/1804.10959)]和BPE-droptout [Provilkov et al](https://arxiv.org/abs/1910.13267)
是实际上简单正则化方法,通过实时子词采样增强了训练数据，这有助于提高NMT模型的准确性和鲁棒性。 

要启用子词正则化，您想集成SentencePiece库
([C++](doc/api.md#sampling-subword-regularization)/[Python](python/README.md)) 
到NMT系统，为每个参数更新采样一个细分，这与标准离线数据准备工作不同。 这是[Python library](python/README.md)的示例。 
您会发现，在每个“SampleEncode(C ++)”或“enable_sampling = True(Python)”编码中，“New York”的细分方式不同。 
采样参数的详细信息可以在 [sentencepiece_processor.h](src/sentencepiece_processor.h).中找到。


```
>>> import sentencepiece as spm
>>> s = spm.SentencePieceProcessor(model_file='spm.model')
>>> for n in range(5):
...     s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest=-1)
...
['▁', 'N', 'e', 'w', '▁York']
['▁', 'New', '▁York']
['▁', 'New', '▁Y', 'o', 'r', 'k']
['▁', 'New', '▁York']
['▁', 'New', '▁York']
```

## Installation

### Python module
SentencePiece提供了同时支持SentencePiece训练和分段的Python装饰器。
您可以使用SentencePiece的Python二进制包进行安装。 

```
% pip install sentencepiece
```

For more detail, see [Python module](python/README.md)

### 从C ++源代码构建和安装SentencePiece命令行工具
需要以下工具和库来构建SentencePiece：

* [cmake](https://cmake.org/)
* C++11 compiler
* [gperftools](https://github.com/gperftools/gperftools) library (可选，可以获得10-40％的性能提升。 )

在Ubuntu上，可以使用apt-get：安装构建工具。 
```
% sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
```

然后，您可以按照以下步骤构建和安装命令行工具:
```
% git clone https://github.com/google/sentencepiece.git 
% cd sentencepiece
% mkdir build
% cd build
% cmake ..
% make -j $(nproc)
% sudo make install
% sudo ldconfig -v
```
On OSX/macOS, 将最后一个命令替换为 `sudo update_dyld_shared_cache`

macos安装5个命令到
```buildoutcfg
-- Installing: /usr/local/bin/spm_encode
-- Installing: /usr/local/bin/spm_decode
-- Installing: /usr/local/bin/spm_normalize
-- Installing: /usr/local/bin/spm_train
-- Installing: /usr/local/bin/spm_export_vocab
```

### Build and install using vcpkg

您可以使用[vcpkg](https://github.com/Microsoft/vcpkg)依赖性管理器下载并安装sentencepiece： 

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install sentencepiece

Microsoft团队成员和社区贡献者不断更新vcpkg中的sentencepiece端口。 如果版本已过期，请在vcpkg repository上[create an issue or pull request](https://github.com/Microsoft/vcpkg)。

## Usage instructions
### 训练 SentencePiece Model
```
% spm_train --input=<input> --model_prefix=<model_name> --vocab_size=8000 --character_coverage=1.0 --model_type=<type>
```
* `--input`: 每行一个句子的原始语料库文件。 无需运行tokenizer,器，标准化器或预处理器。 默认情况下，SentencePiece使用Unicode NFKC标准化输入。 您可以传递以逗号分隔的文件列表。 
* `--model_prefix`: 输出模型名称前缀. `<model_name>.model` and `<model_name>.vocab` are generated.
* `--vocab_size`: 单词表大小, e.g., 8000, 16000, or 32000
* `--character_coverage`: 该模型覆盖的字符数，默认值为：日语字符或中文字符丰富的语言为“0.9995”，其他字符较小的语言的字符为“1.0”。  
* `--model_type`: 模型类型。 从“unigram”(默认)，“bpe”，“char”或“word”中选择。 使用`word`类型时，必须对输入句子进行pretokenized。 

使用`--help`标志来显示所有训练参数，或查看[here](doc/options.md) 以获取概述。


### 将原始文本编码为句子片段/id 
```
% spm_encode --model=<model_file> --output_format=piece < input > output
% spm_encode --model=<model_file> --output_format=id < input > output
```

使用`--extra_options`标志来插入BOS/EOS标记或反转输入序列。 
```
% spm_encode --extra_options=eos (add </s> only)
% spm_encode --extra_options=bos:eos (add <s> and </s>)
% spm_encode --extra_options=reverse:bos:eos (reverse input and add <s> and </s>)
```

SentencePiece支持带有`--output_format =(nbest | sample)_(piece | id)`标志的nbest分段和分段采样。 
```
% spm_encode --model=<model_file> --output_format=sample_piece --nbest_size=-1 --alpha=0.5 < input > output
% spm_encode --model=<model_file> --output_format=nbest_id --nbest_size=10 < input > output
```

### 将句子片段/id解码为原始文本 
```
% spm_decode --model=<model_file> --input_format=piece < input > output
% spm_decode --model=<model_file> --input_format=id < input > output
```
使用`--extra_options`标志以相反的顺序解码文本。 
```
% spm_decode --extra_options=reverse < input > output
```

### End-to-End Example端到端的示例
```
% spm_train --input=data/botchan.txt --model_prefix=m --vocab_size=1000
unigram_model_trainer.cc(494) LOG(INFO) Starts training with :
input: "../data/botchan.txt"
... <snip>
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=1 size=1100 obj=10.4973 num_tokens=37630 num_tokens/piece=34.2091
trainer_interface.cc(272) LOG(INFO) Saving model: m.model
trainer_interface.cc(281) LOG(INFO) Saving vocabs: m.vocab

% echo "I saw a girl with a telescope." | spm_encode --model=m.model
▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe .

% echo "I saw a girl with a telescope." | spm_encode --model=m.model --output_format=id
9 459 11 939 44 11 4 142 82 8 28 21 132 6

% echo "9 459 11 939 44 11 4 142 82 8 28 21 132 6" | spm_decode --model=m.model --input_format=id
I saw a girl with a telescope.
```
您会发现原始输入句子已从单词表ID序列中恢复。 

### Export vocabulary list
```
% spm_export_vocab --model=<model_file> --output=<output file>
```
```<output file>``` 存储单词表和emission日志概率列表。 单词表ID与此文件中的行号相对应。 

### Redefine special meta tokens
默认情况下， SentencePiece 使用未知(&lt;unk&gt;), BOS (&lt;s&gt;) and EOS (&lt;/s&gt;) token，其具有 的 0 ，1和 2 的ID。 我们可以在训练阶段重新定义此映射，如下所示。

```
% spm_train --bos_id=0 --eos_id=1 --unk_id=5 --input=... --model_prefix=... --character_coverage=...
```

当设置-1 id时，例如```bos_id = -1```，此特殊token被禁用。 请注意，无法禁用unknow id。 我们可以将padding(&lt;pad&gt;) 的ID定义为```--pad_id = 3```。 
 

如果要分配其他特殊token，请参阅[Use custom symbols](doc/special_symbols.md)。


### Vocabulary restriction

spm_encode接受一个--vocabulary和一个vocabulary_threshold选项，
以便spm_encode产生会出现在单词表中的符号(至少有一些频率)。 
[subword-nmt page](https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt).中介绍了此特征的背景。

用法基本上与`subword-nmt`相同。 假设L1和L2是两种语言(源/目标语言)，训练共享的spm模型，并获得每种语言的最终单词表量：

```
% cat {train_file}.L1 {train_file}.L2 | shuffle > train
% spm_train --input=train --model_prefix=spm --vocab_size=8000 --character_coverage=0.9995
% spm_encode --model=spm.model --generate_vocabulary < {train_file}.L1 > {vocab_file}.L1
% spm_encode --model=spm.model --generate_vocabulary < {train_file}.L2 > {vocab_file}.L2
```

```shuffle``` 命令是为了以防万一，因为默认情况下`spm_train`加载语料库的前10M行。 


Then segment train/test corpus with ```--vocabulary``` option
```
% spm_encode --model=spm.model --vocabulary={vocab_file}.L1 --vocabulary_threshold=50 < {test_file}.L1 > {test_file}.seg.L1
% spm_encode --model=spm.model --vocabulary={vocab_file}.L2 --vocabulary_threshold=50 < {test_file}.L2 > {test_file}.seg.L2
```

## Advanced topics

* [SentencePiece Experiments](doc/experiments.md)
* [SentencePieceProcessor C++ API](doc/api.md)
* [Use custom text normalization rules](doc/normalization.md)
* [Use custom symbols](doc/special_symbols.md)
* [Python Module](python/README.md)
* [TensorFlow Module](tensorflow/README.md)
* [Segmentation and training algorithms in detail]

