# SentencePiece Python Wrapper

SentencePiece的Python装饰器。 该API将提供Sentencepiece的编码，解码和训练。

## Build and Install SentencePiece
For Linux (x64/i686), macOS, and Windows(win32/x64) environment,您只需使用pip命令安装SentencePiece python模块即可。 

```
% pip install sentencepiece
```

从源代码构建和安装Python装饰器 , please install [SentencePiece C++](https://github.com/google/sentencepiece#c-from-source) and try the following commands:
```
% python setup.py build
% sudo python setup.py install
```

如果您没有对global site-packages包目录的写许可权或不想安装到该目录中，请尝试：
```
% python setup.py install --user
```

## Usage

See [this google colab page](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb) to run sentencepiece interactively. (Note: this sample is written in old interface.)

### Segmentation
```
% python
>>> import sentencepiece as spm
>>> sp = spm.SentencePieceProcessor(model_file='test/test_model.model')
>>> sp.encode('This is a test')
[284, 47, 11, 4, 15, 400]
>>> sp.encode(['This is a test', 'Hello world'], out_type=int)
[[284, 47, 11, 4, 15, 400], [151, 88, 21, 887]]
>>> sp.encode('This is a test', out_type=str)
['▁This', '▁is', '▁a', '▁', 't', 'est']
>>> sp.encode(['This is a test', 'Hello world'], out_type=str)
[['▁This', '▁is', '▁a', '▁', 't', 'est'], ['▁He', 'll', 'o', '▁world']]
>>> for _ in range(10):
...     sp.encode('This is a test', out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
... 
['▁', 'This', '▁', 'is', '▁a', '▁', 't', 'e', 'st']
['▁T', 'h', 'i', 's', '▁is', '▁a', '▁', 'te', 's', 't']
['▁T', 'h', 'is', '▁', 'is', '▁', 'a', '▁', 't', 'est']
['▁', 'This', '▁is', '▁', 'a', '▁', 't', 'e', 'st']
['▁', 'This', '▁', 'is', '▁', 'a', '▁', 't', 'e', 's', 't']
['▁This', '▁is', '▁a', '▁', 'te', 's', 't']
['▁This', '▁is', '▁', 'a', '▁', 't', 'e', 'st']
['▁', 'T', 'h', 'is', '▁', 'is', '▁', 'a', '▁', 'te', 'st']
['▁', 'This', '▁', 'i', 's', '▁a', '▁', 't', 'e', 'st']
['▁This', '▁', 'is', '▁a', '▁', 't', 'est']
>>> sp.decode([284, 47, 11, 4, 15, 400])
'This is a test'
>>> sp.decode([[284, 47, 11, 4, 15, 400], [151, 88, 21, 887]])
['This is a test', 'Hello world']
>>> sp.decode(['▁', 'This', '▁', 'is', '▁a', '▁', 't', 'e', 'st'])
'This is a test'
>>> sp.decode([['▁This', '▁is', '▁a', '▁', 't', 'est'], ['▁He', 'll', 'o', '▁world']])
['This is a test', 'Hello world']
>>> sp.get_piece_size()
1000
>>> sp.id_to_piece(2)
'</s>'
>>> sp.id_to_piece([2, 3, 4])
['</s>', '\r', '▁']
>>> sp.piece_to_id('<s>')
1
>>> sp.piece_to_id(['</s>', '\r', '▁'])
[2, 3, 4]
>>> len(sp)
1000
>>> sp['</s>']
2
```

### Model Training
通过将[spm_train](https://github.com/google/sentencepiece#train-sentencepiece-model)的参数传递给SentencePieceTrainer.train()函数来进行训练。

```
>>> import sentencepiece as spm
>>> spm.SentencePieceTrainer.train(input='test/botchan.txt', model_prefix='m', vocab_size=1000, user_defined_symbols=['foo', 'bar'])
sentencepiece_trainer.cc(73) LOG(INFO) Starts training with : 
trainer_spec {
  input: test/botchan.txt
  .. snip
unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=1 size=1188 obj=10.2839 num_tokens=32182 num_tokens/piece=27.0892
unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=0 size=1100 obj=10.4269 num_tokens=33001 num_tokens/piece=30.0009
unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=1 size=1100 obj=10.4069 num_tokens=33002 num_tokens/piece=30.0018
trainer_interface.cc(595) LOG(INFO) Saving model: m.model
trainer_interface.cc(619) LOG(INFO) Saving vocabs: m.vocab
>>>
```

### Training without local filesystem
Sentencepiece训练器可以接收任何可迭代的对象来输入训练语句。 
您还可以传递文件对象(带有write()方法的实例)以将输出模型发送给任何设备。 
这些对于在对局部文件系统(例如Google colab)具有有限访问权限的环境中运行句子非常有用。 
```
import urllib.request
import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()
with urllib.request.urlopen(
    'https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt'
) as response:
  spm.SentencePieceTrainer.train(
      sentence_iterator=response, model_writer=model, vocab_size=1000)

# Serialize the model as file.
# with open('out.model', 'wb') as f:
#   f.write(model.getvalue())

# Directly load the model from serialized model.
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
print(sp.encode('this is test'))
```


### Segmentation (old interface)
```
% python
>>> import sentencepiece as spm
>>> sp = spm.SentencePieceProcessor()
>>> sp.Load("test/test_model.model")
True
>>> sp.EncodeAsPieces("This is a test")
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est']
>>> sp.EncodeAsIds("This is a test")
[284, 47, 11, 4, 15, 400]
>>> sp.DecodePieces(['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est'])
'This is a test'
>>> sp.NBestEncodeAsPieces("This is a test", 5)
[['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 'st'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'es', 't']]
>>> for x in range(10):
...     sp.SampleEncodeAsPieces("This is a test", -1, 0.1)
...
['\xe2\x96\x81', 'T', 'h', 'i', 's', '\xe2\x96\x81', 'is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 's', 't']
['\xe2\x96\x81T', 'h', 'is', '\xe2\x96\x81is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'est']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 's', 't']
['\xe2\x96\x81T', 'h', 'is', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't']
['\xe2\x96\x81This', '\xe2\x96\x81', 'is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't']
['\xe2\x96\x81This', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81', 'is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 'te', 's', 't']
>>> sp.DecodeIds([284, 47, 11, 4, 15, 400])
'This is a test'
>>> sp.GetPieceSize()
1000
>>> sp.IdToPiece(2)
'</s>'
>>> sp.PieceToId('</s>')
2
>>> len(sp)
1000
>>> sp['</s>']
2
```

### Model Training (old interface)
Training is performed by passing parameters of [spm_train](https://github.com/google/sentencepiece#train-sentencepiece-model) to  SentencePieceTrainer.Train() function.

```
>>> import sentencepiece as spm
>>> spm.SentencePieceTrainer.Train('--input=test/botchan.txt --model_prefix=m --vocab_size=1000')
unigram_model_trainer.cc(494) LOG(INFO) Starts training with : 
input: "test/botchan.txt"
model_prefix: "m"
model_type: UNIGRAM
..snip..
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=0 size=1239 obj=10.4055 num_tokens=36256 num_tokens/piece=29.2623
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=1 size=1239 obj=10.3187 num_tokens=36256 num_tokens/piece=29.2623
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=0 size=1100 obj=10.5285 num_tokens=37633 num_tokens/piece=34.2118
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=1 size=1100 obj=10.4973 num_tokens=37630 num_tokens/piece=34.2091
trainer_interface.cc(284) LOG(INFO) Saving model: m.model
trainer_interface.cc(293) LOG(INFO) Saving vocabs: m.vocab
>>>
```

## Python2/3 String/Unicode compatibility
Sentencepiece python装饰器接受Unicode字符串和旧字节字符串。
输出字符串类型由输入字符串类型确定。
IdToPiece/DecodeIds方法的输出类型为*str*，但请注意，它分别是Python2中的旧字节字符串和Python3中的Unicode字符串。 

* Python2:
```
>>> sp.EncodeAsPieces('吾輩は猫である')
['\xe2\x96\x81', '\xe5\x90\xbe', '\xe8\xbc\xa9', '\xe3\x81\xaf', '\xe7\x8c\xab', '\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>> sp.EncodeAsPieces(u'吾輩は猫である')
[u'\u2581', u'\u543e', u'\u8f29', u'\u306f', u'\u732b', u'\u3067\u3042\u308b']
>>> sp.EncodeAsPieces(u'吾輩は猫である'.encode('utf-8'))
['\xe2\x96\x81', '\xe5\x90\xbe', '\xe8\xbc\xa9', '\xe3\x81\xaf', '\xe7\x8c\xab', '\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>> sp.IdToPiece(10)
'\xe3\x81\xab'
>>> type(sp.IdToPiece(10))
<type 'str'>
```

* Python3:
```
>>> sp.EncodeAsPieces('吾輩は猫である')
['▁', '吾', '輩', 'は', '猫', 'である']
>>> sp.EncodeAsPieces('吾輩は猫である'.encode('utf-8'))
[b'\xe2\x96\x81', b'\xe5\x90\xbe', b'\xe8\xbc\xa9', b'\xe3\x81\xaf', b'\xe7\x8c\xab', b'\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>>
>>> sp.IdToPiece(10)
'に'
>>> type(sp.IdToPiece(10))
<class 'str'>
```
