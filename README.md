# korean-conversation
한국어 대화를 진행하는 인공지능을 개발하고자 시도 중인 취미 프로젝트입니다.

### 21.10.12
POSTagger 뿐아니라 문장의 주요소를 태깅하여 쉽게 정보를 뽑아오도록 하려고 STRUCTagger를 만드는 중입니다.    
LSTM을 이용하여 양방향 태깅을 위한 시퀀스 레이블링을 따라 구현해보았습니다.    
참고한 사이트는 [02. 양방향 RNN을 이용한 품사 태깅](https://wikidocs.net/66747)과 [사용자 정의 DATASET, DATALOADER, TRANSFORMS 작성하기](https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html)입니다.    
첫 사이트에서는 품사 태깅을 할 때 어떤 식으로 인공지능이 작성되는 지에 대한 정보를 얻기 위함이고, 두번째 사이트에서는 직접 만든 데이터 셋을 어떻게 불러오고, 배치사이즈를 정하고 사용할 준비를 하기 위한 정보를 얻기 위함입니다.    
따라서 [StructrueDataset](https://github.com/HanGyeolee/korean-conversation/blob/main/structure/structure.py#L25)은 두번째 사이트를 보고 구현하였고, [RNNSTRUCTagger](https://github.com/HanGyeolee/korean-conversation/blob/main/structure/structure.py#L64)는 첫번째 사이트를 보고 따라한 것 입니다.    

주요소에 사용되는 태그들은 다음과 같습니다.
|서술어|주어|언제|왜|어디에서|어디까지|직접목적어|간접목적어|어떻게|누구와|./?/!|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|V|S|T|Wy|WS|WE|DO|IO|H|Wi|EOF|

문장 속 형태소 최대 개수는 512개로 제한하였습니다.    
데이터 셋을 만들다가 부족하다싶으면 더 늘려볼 생각입니다.

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
