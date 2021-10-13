# korean-conversation
한국어 대화를 진행하는 인공지능을 개발하고자 시도 중인 개인 오픈소스 프로젝트입니다.   

학습에 사용된 컴퓨터의 성능은 다음과 같습니다.

|부품|이름|
|:---:|:---|
|CPU|AMD Ryzen 7 5800X 8-Core Processor 3.80 GHz|
|GPU|NVIDIA GeForce RTX 3070 8G|
|RAM|GSkill DDR4-3600 16GB * 2|

## Table
* [Update](#update)

## Update
### 21.10.13
[ELEMENTagger](https://github.com/HanGyeolee/korean-conversation/blob/main/mecab/ko/elemen.py)는 문장의 주요소를 추가로 태깅해줍니다.    
``` python
from mecab.ko.tokenizer import Tokenizer
from mecab.ko.elemen import ELEMENTagger

tokenizer = Tokenizer(dicpath=r'vocab.txt', update=False) 
tokenizer.tokenizing(string, allattrs=False) #토큰 생성

elementagger = ELEMENTagger(ptpath="~~.pt")
elementagger.getElement(tokenizer.tokens) #주요소 추출
```

[공식 튜토리얼 사이트](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)를 참고하여 학습기를 완성시켰습니다.    
아직 데이터셋이 부족하여, Loss 가 2 이상 발생하는 중입니다. (말뭉치를 요청하여 데이터 구축을 준비하는 중입니다.)    

|항목|데이터셋 예시|
|:---:|:---|
|text| 	&#91;'밖/NNG', '에/JKB', '비오/VV', 'ㅏ/EF', '?/SF'&#93; |
|element|	&#91;'WS', 'WS', 'V', 'V', 'EOF'&#93;|
|start|	&#91;1, 1, 3, 3, 5&#93;|
|end|	&#91;2, 2, 4, 4, 5&#93;|
|length|5|

### 21.10.12
POSTagger 뿐아니라 문장의 주요소를 태깅하여 쉽게 정보를 뽑아오도록 하려고 ELEMENTagger를 만드는 중입니다.    
LSTM을 이용하여 양방향 태깅을 위한 시퀀스 레이블링을 따라 구현해보았습니다.    
참고한 사이트는 [02. 양방향 RNN을 이용한 품사 태깅](https://wikidocs.net/66747)과 [사용자 정의 DATASET, DATALOADER, TRANSFORMS 작성하기](https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html)입니다.    

주요소에 사용되는 태그들은 다음과 같습니다.
* 중요도 높음

|V|S|T|Wy|
|:---:|:---:|:---:|:---:|
|서술어|주어|시간|이유|
    
* 중요도 낮음  
 
|WS|WE|DO|IO|H|Wi|EOF|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|어디에서|어디로|직접목적어|간접목적어|어떻게|누구와|./?/!|

문장 속 형태소 최대 개수는 512개로 제한하였습니다.    
데이터 셋을 만들다가 부족하다싶으면 더 늘려볼 생각입니다.
<hr/>

### 21.10.10
형태소를 뽑아내는 데에 [KoNLPy](https://konlpy.org/ko/latest/) Mecab 클래스를 사용하였습니다.    
mecab 클래스를 약간 수정하여 "+" 된 어절을 전부 분리하도록 하였습니다.    
[mecab/mecab_tokenizer.py](https://github.com/HanGyeolee/korean-conversation/blob/main/mecab/mecab_tokenizer.py)를 통해 분리된 형태소들을 토크나이징할 수 있습니다.    
숫자는 ${number}/SN 로 토크나이징 될 것입니다.

mecab-ko-dic의 모든 어절을 vocab.txt에 저장해두었습니다.    
vocab.txt에 들어간 전체 어휘의 개수는 816288개 입니다.

<b>Ex)</b>    
> 경기 성남시 판교신도시에서 이달 분양하는 중대형 아파트의 3.3m²당 분양가가 2006년보다 200만 원 정도 싼 1500만 원 후반대로 결정될 것으로 보인다.

``` json
[
  218296, 724373, 317499, 396668, 812437, 254248, 191715, 350259, 249027, 291460, 
  816276, 4587, 372880, 816256, 10582, 191764, 816287, 816288, 816287, -1, 
  -1, 816164, 291460, 816137, 191481, 816287, 207053, 191664, 816287, 418926, 
  207370, 365195, 767213, 816284, 816287, 418926, 207370, 413800, 191548, 217693, 
  816266, 816285, 206811, 191746, 771756, 816286, 765791
]
```
-1은 vocab.txt에 포함되지 않은 단어들입니다. 위의 예시에서는 m와 ²가 각각 -1 값을 반환하는 데, 해당 데이터를 무작정 넣어봐야하는 지 고민입니다.    
[test.py의 result](https://github.com/HanGyeolee/korean-conversation/blob/main/test.py#L38) 값에 따라 반환되는 형태를 변경할 수 있으니 여러가지로 활용해볼 수 있겠습니다.
