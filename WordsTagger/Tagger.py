from __future__ import absolute_import

import pathlib
import MeCab
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from typing_extensions import TypedDict

Cuda = True


# 해당 태거는 문장을 띄어쓰기에 맞춰 나누고, 나눠진 어절들을 형태소로 분리해서 어절끼리 LSTM 학습한 후
# 학습된 데이터를 다시 합쳐 문장으로 LSTM 학습하는 과정을 거친다.

class MorpToEojeolEmbedding(nn.Module):
    def __init__(self, n_morp_vocab, d_morp_embed, d_eojeol_embed, n_layers, dropout_p=0.2):
        """
        n_morp_vacab : 형태소 / 형태소 태그
        d_morp_embed : 형태소 임베딩 / 형태소 태그 임베딩
        d_eojeol_embed : 어절 임베딩
        """
        super(MorpToEojeolEmbedding, self).__init__()

        self.n_morp_vocab = n_morp_vocab
        self.d_morp_embed = d_morp_embed
        self.d_eojeol_embed = d_eojeol_embed
        self.n_layers = n_layers

        self.embed = nn.Embedding(n_morp_vocab, d_morp_embed)
        self.gru = nn.GRU(d_morp_embed, d_eojeol_embed, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def set_embedded_param(self, weight_matrix):
        self.embed.weight = torch.nn.Parameter(torch.from_numpy(weight_matrix))

    def get_embedded_param(self):
        return self.embed.weight

    def forward(self, x):
        # x = (어절, 형태소)
        x_morp_embed = self.embed(x)

        n_seq = x_morp_embed.size(0)
        h0 = self._init_state(n_seq)
        h, _ = self.gru(x_morp_embed, h0)
        ht = h[:, -1, :]
        self.dropout(ht)

        return ht

    def _init_state(self, n_seq=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, n_seq, self.d_eojeol_embed).zero_()


class BidirectionalLSTMEmbedding(nn.Module):
    def __init__(self, d_eojeol, d_lstm_embed, n_layers):
        """
        d_eojeol : 어절 임베딩
        d_lstm_embed : 결과
        """
        super(BidirectionalLSTMEmbedding, self).__init__()

        self.d_eojeol = d_eojeol
        self.d_lstm_embed = d_lstm_embed
        self.n_layers = n_layers

        self.blstm = nn.LSTM(d_eojeol, d_lstm_embed, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.unsqueeze(0)
        h0, c0 = self._init_state()
        ht, _ = self.blstm(x, (h0, c0))

        return ht.squeeze(0)

    def _init_state(self):
        n_batchs = 1
        weight = next(self.parameters()).data
        h0 = weight.new(self.n_layers * 2, n_batchs, self.d_lstm_embed).zero_()
        c0 = weight.new(self.n_layers * 2, n_batchs, self.d_lstm_embed).zero_()
        return h0, c0


class MLP(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers=2, dropout=0.3, activation='relu'):
        super(MLP, self).__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.n_layers = n_layers
        self.drop = nn.Dropout(dropout)

        self.MLP = nn.ModuleList()
        if n_layers == 1:
            self.MLP.append(nn.Linear(d_input, d_output))
        else:
            self.MLP.append(nn.Linear(d_input, d_hidden))
            for _ in range(n_layers - 2):
                self.MLP.append(nn.Linear(d_hidden, d_hidden))
            self.MLP.append(nn.Linear(d_hidden, d_output))

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x
        for i in range(self.n_layers - 1):
            out = self.MLP[i](self.drop(out))
            out = self.activation(out)
        return self.MLP[-1](self.drop(out))


class WordTagger(nn.Module):
    def __init__(self, n_morp_lemmas, d_morp_lemma_embed, d_eojeol_lemma_embed,
                 n_morp_poses, d_morp_pos_embed, d_eojeol_pos_embed,
                 d_lstm_hidden, n_lstm_layers,
                 d_mlp_hidden, n_mlp_layers,
                 d_mlp_eojeol_pos_output,
                 p_mlp_dropout):
        super(WordTagger, self).__init__()

        self.n_morp_lemmas = n_morp_lemmas
        self.d_morp_lemma_embed = d_morp_lemma_embed
        self.d_eojeol_lemma_embed = d_eojeol_lemma_embed
        self.n_morp_poses = n_morp_poses
        self.d_morp_pos_embed = d_morp_pos_embed
        self.d_eojeol_pos_embed = d_eojeol_pos_embed
        self.d_lstm_hidden = d_lstm_hidden
        self.n_lstm_layers = n_lstm_layers
        self.d_mlp_hidden = d_mlp_hidden
        self.n_mlp_layers = n_mlp_layers
        self.d_mlp_output = d_mlp_eojeol_pos_output

        self.morp_lemma_to_eojeol_embed = MorpToEojeolEmbedding(n_morp_lemmas, d_morp_lemma_embed, d_eojeol_lemma_embed,
                                                                1)
        self.morp_pos_to_eojeol_embed = MorpToEojeolEmbedding(n_morp_poses, d_morp_pos_embed, d_eojeol_pos_embed, 1)
        self.bidirectional_lstm_embed = BidirectionalLSTMEmbedding(d_eojeol_lemma_embed + d_eojeol_pos_embed,
                                                                   d_lstm_hidden, n_lstm_layers)

        self.mlp_eojeol_pos = MLP(d_lstm_hidden * 2, d_mlp_hidden, d_mlp_eojeol_pos_output, n_mlp_layers,
                                  p_mlp_dropout, 'tanh')

    def set_embedded_param(self, lemma_weight_matrix, pos_weight_matrix):
        self.morp_lemma_to_eojeol_embed.set_embedded_param(lemma_weight_matrix)
        self.morp_pos_to_eojeol_embed.set_embedded_param(pos_weight_matrix)

    def get_embedded_param(self):
        return self.morp_lemma_to_eojeol_embed.get_embedded_param(), self.morp_pos_to_eojeol_embed.get_embedded_param()

    def forward(self, morp_lemmas, morp_poses):
        eojeol_embed_morp_lemma = self.morp_lemma_to_eojeol_embed(morp_lemmas)
        eojeol_embed_morp_pos = self.morp_pos_to_eojeol_embed(morp_poses)
        eojeol_embed_morp = torch.cat([eojeol_embed_morp_lemma, eojeol_embed_morp_pos], axis=-1)

        eojeol_embed_lstm = self.bidirectional_lstm_embed(eojeol_embed_morp)
        eojeol_pos_scores = self.mlp_eojeol_pos(eojeol_embed_lstm)

        return eojeol_pos_scores


def parse(result, allattrs=False, join=False):
    def split(elem, join=False):
        if not elem: return ('', 'SY')
        s, t = elem.split('\t')
        if join:
            splited = t.split(',')
            return s + '/' + (splited[0] if ('+' not in splited[7]) or (splited[4] != "Inflect") else splited[7])
        else:
            splited = t.split(',')
            return (s, splited[0]) if ('+' not in splited[7]) or (splited[4] != "Inflect") else (s, splited[7])

    def attrs(elem, join=False):
        if not elem: return ('', 'SY')
        s, t = elem.split('\t')
        if join:
            return s + '/' + t
        else:
            return (s, t)

    if allattrs:
        return [attrs(elem, join=join) for elem in result.splitlines()[:-1]]
    return [split(elem, join=join) for elem in result.splitlines()[:-1]]


tag_to_ix = {"EOS": 0, "VP": 1, "SBJ": 2, "OBJ": 3, "AJT": 4, "MOD": 5, "CMP": 6, "CMJ": 7}
ix_to_tag = {0: "EOS", 1: "VP", 2: "SBJ", 3: "OBJ", 4: "AJT", 5: "MOD", 6: "CMP", 7: "CMJ"}


def getstructure(predict):
    structure = predict.tolist()
    for i, sub in enumerate(structure):
        structure[i] = sub.index(max(sub))

    return [(ix_to_tag[v] if v in ix_to_tag else None) for v in structure]


mecab = MeCab.Tagger()


class Embedding(TypedDict):
    words: np.ndarray
    vectors: np.ndarray


def load_embedding(embedding_filepath: pathlib.Path) -> Embedding:
    with open(embedding_filepath, 'r', encoding='utf-8') as fp:
        n_morp_lemma, d_morp_lemma = fp.readline().strip().split(' ')
        n_morp_lemma = int(n_morp_lemma)
        d_morp_lemma = int(d_morp_lemma)
        lines = fp.readlines()

    words = []
    vectors = []

    for line in lines:
        word_vector = line.rstrip().rsplit(' ', d_morp_lemma)
        word = word_vector[0]
        vector = word_vector[1:]

        words.append(word)
        vectors.append(vector)

    words = np.array(words, dtype=str)
    vectors = np.array(vectors, dtype=float)

    return dict(words=words, vectors=vectors)


class Sentence(TypedDict):
    form: str
    eojeols: List[dict]
    poses: List[str]


def modify_dataset(corpus: List[Sentence]):
    for sent in corpus:
        sent["eojeols"] = []
        for eojeol in sent["form"].split():
            tmp = {
                "morp_lemmas": [],
                "morp_poses": []
            }
            parsed = mecab.parse(eojeol)
            """ 한 어절 """
            for splited in parsed.split('\n'):
                if splited == "EOS":
                    break
                morp_form, morp_pos = splited.rsplit('\t', 1)
                morp_pos = morp_pos.split(',')[0]
                if morp_pos[0] == 'V':
                    morp_form += '다'
                tmp["morp_lemmas"].append(morp_form)
                tmp["morp_poses"].append(morp_pos)
            sent["eojeols"].append(tmp)

    print(corpus)


def get_train_data(corpus, bos_token, eos_token, num_token) -> Tuple[
    List[List[List[str]]], List[List[List[str]]], List[List[str]]]:
    """
    Corpus에서 학습에 사용할 데이터만 추출

    """
    morp_lemmas = [[
        [bos_token] + [num_token if lemma.isnumeric() else lemma for lemma in eojeol['morp_lemmas']] + [eos_token]
        for eojeol in sent["eojeols"]] for sent in corpus]
    morp_poses = [[
        [bos_token] + eojeol['morp_poses'] + [eos_token] for eojeol in sent["eojeols"]] for sent in corpus]

    eojeol_poses = [[pos for pos in sent["poses"]] for sent in corpus]

    return morp_lemmas, morp_poses, eojeol_poses


class Vocab(TypedDict):
    stoi: dict
    itos: dict


def get_vocabs(morp_lemmas, morp_poses, eojeol_poses):
    # Vocab 생성: Dictionary stoi, itos 생성
    def get_vocab(train_data: List[List[str]]) -> Vocab:
        field = torchtext.data.Field(sequential=True, batch_first=True, lower=True)
        field.build_vocab(train_data)

        return dict(
            stoi=field.vocab.stoi,
            itos=field.vocab.itos
        )

    if embedding is None:
        morp_lemma_vocab = dict(
            stoi={w: i for i, w in enumerate(embedding['words'])},
            itos={i: w for i, w in enumerate(embedding['words'])}
        )
    else:
        morp_lemma_vocab = get_vocab(morp_lemmas)

    morp_pos_vocab = get_vocab(morp_poses)
    eojeol_pos_vocab = get_vocab(eojeol_poses)

    return morp_lemma_vocab, morp_pos_vocab, eojeol_pos_vocab


token = {"bos_token": "BOS", "eos_token": "EOS", "num_token": "NUM"}

dataset = [
    ["나는 친구와 밥을 집에서 먹었다.",  ["SBJ", "AJT", "OBJ", "AJT", "VP"]],
    ["나는 밥을 친구와 집에서 먹었다.", ["SBJ", "OBJ", "AJT", "AJT", "VP"]],
    ["나는 집에서 밥을 친구와 먹었다.",  ["SBJ", "AJT", "OBJ", "AJT", "VP"]],
    ["친구와 나는 밥을 집에서 먹었다.", ["AJT", "SBJ", "OBJ", "AJT", "VP"]],
    ["나는 밥을 집에서 먹었다.", ["SBJ", "OBJ", "AJT", "VP"]],
    ["친구는 밥을 집에서 먹었다.", ["SBJ", "OBJ", "AJT", "VP"]],
    ["나는 집에서 밥을 먹었다.", ["SBJ", "AJT", "OBJ", "VP"]],
    ["나는 친구와 밥을 먹었다.", ["SBJ", "AJT", "OBJ", "VP"]],
    ["친구는 집에서 밥을 먹었다.", ["SBJ", "AJT", "OBJ", "VP"]],
    ["나는 밥을 먹었다.", ["SBJ", "OBJ", "VP"]],
    ["친구는 밥을 먹었다.", ["SBJ", "OBJ", "VP"]],
]

corpus: List[Sentence] = [{"form": line[0], "poses": line[1]} for line in dataset]
print(corpus)

embedding_filepath = "WordEmbedding.txt"
if type(embedding_filepath) == str:
    embedding_filepath = pathlib.Path(embedding_filepath)

modify_dataset(corpus)
embedding = load_embedding(embedding_filepath)

morp_lemmas, morp_poses, eojeol_poses = get_train_data(
    corpus, token["bos_token"], token["eos_token"], token["num_token"])
morp_lemma_vocab, morp_pos_vocab, eojeol_pos_vocab = get_vocabs(
    morp_lemmas, morp_poses, eojeol_poses)