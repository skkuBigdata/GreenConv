# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
from __future__ import print_function

import six
from six.moves import map

from metric.pycocoevalcap.bleu.bleu import Bleu
from metric.pycocoevalcap.cider.cider import Cider
from metric.pycocoevalcap.meteor.meteor import Meteor
from metric.pycocoevalcap.rouge.rouge import Rouge


# str/unicode stripping in Python 2 and 3 instead of `str.strip`.
def _strip(s):
    return s.strip()


def compute_metrics(hypothesis, references, no_overlap=False, no_glove=False):
    with open(hypothesis, 'r') as f:
        hyp_list = f.readlines()
    ref_list = []
    for iidx, reference in enumerate(references):
        with open(reference, 'r') as f:
            ref_list.append(f.readlines())
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        print("no_overlap.")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                print("%s: %0.6f" % (method, score))
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    if not no_glove:
        print("53 no_glove.")
        from metric.word2vec.evaluate import eval_emb_metrics
        import numpy as np

        glove_hyps = [h.strip() for h in hyp_list]
        ref_list_T = np.array(ref_list).T.tolist()
        glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
        scores, scores_list_dict = eval_emb_metrics(glove_hyps, glove_refs)
        print(scores)
        scores = scores.split('\n')
        for score in scores:
            name, value = score.split(':')
            value = float(value.strip())
            ret_scores[name] = value

    return ret_scores


def compute_individual_metrics(ref, hyp, no_overlap=False, no_glove=False):
    assert isinstance(hyp, six.string_types)

    if isinstance(ref, six.string_types):
        ref = ref.split('||<|>||')  # special delimiter for backward compatibility
    ref = [a.strip() for a in ref]
    refs = {0: ref}
    ref_list = [ref]

    hyps = {0: [hyp.strip()]}
    hyp_list = [hyp]

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc
            else:
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    if not no_glove:
        print("103 no_glove.")
        from metric.word2vec.evaluate import eval_emb_metrics
        import numpy as np

        glove_hyps = [h.strip() for h in hyp_list]
        ref_list_T = np.array(ref_list).T.tolist()
        glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
        scores, scores_list_dict = eval_emb_metrics(glove_hyps, glove_refs)
        scores = scores.split('\n')
        for score in scores:
            name, value = score.split(':')
            value = float(value.strip())
            ret_scores[name] = value

    return ret_scores


class NLGEval(object):
    glove_metrics = {
        'EmbeddingAverageCosineSimilarity',
        'VectorExtremaCosineSimilarity',
        'GreedyMatchingScore',
    }

    valid_metrics = {
                        # Overlap
                        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                        'METEOR',
                        'ROUGE_L',
                        'CIDEr',

                        # Skip-thought
                        'SkipThoughtCS',
                    } | glove_metrics

    def __init__(self, no_overlap=False, no_glove=False,
                 metrics_to_omit=None):
        """
        :param no_overlap: Default: Use overlap metrics.
            `True` if these metrics should not be used.
        :type no_overlap: bool
        :param no_glove: Default: Use GloVe based metrics.
            `True` if these metrics should not be used.
        :type no_glove: bool
        :param metrics_to_omit: Default: Use all metrics. See `NLGEval.valid_metrics` for all metrics.
            The previous parameters will override metrics in this one if they are set.
            Metrics to omit. Omitting Bleu_{i} will omit Bleu_{j} for j>=i.
        :type metrics_to_omit: Optional[Collection[str]]
        """

        if metrics_to_omit is None:
            self.metrics_to_omit = set()
        else:
            self.metrics_to_omit = set(metrics_to_omit)
            # For backwards compatibility.
            if 'EmbeddingAverageCosineSimilairty' in self.metrics_to_omit:
                self.metrics_to_omit.remove('EmbeddingAverageCosineSimilairty')
                self.metrics_to_omit.add('EmbeddingAverageCosineSimilarity')

        assert len(self.metrics_to_omit - self.valid_metrics) == 0, \
            "Invalid metrics to omit: {}".format(self.metrics_to_omit - self.valid_metrics)

        self.no_overlap = no_overlap
        if not no_overlap:
            self.load_scorers()

        self.no_glove = no_glove or len(self.glove_metrics - self.metrics_to_omit) == 0
        if not self.no_glove:
            self.load_glove()

    def load_scorers(self):
        self.scorers = []

        omit_bleu_i = False
        for i in range(1, 4 + 1):
            if 'Bleu_{}'.format(i) in self.metrics_to_omit:
                omit_bleu_i = True
                if i > 1:
                    self.scorers.append((Bleu(i - 1), ['Bleu_{}'.format(j) for j in range(1, i)]))
                break
        if not omit_bleu_i:
            self.scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))

        if 'METEOR' not in self.metrics_to_omit:
            self.scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' not in self.metrics_to_omit:
            self.scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' not in self.metrics_to_omit:
            self.scorers.append((Cider(), "CIDEr"))

    def load_glove(self):
        from metric.word2vec.evaluate import Embedding
        from metric.word2vec.evaluate import eval_emb_metrics
        import numpy as np
        self.eval_emb_metrics = eval_emb_metrics
        self.np = np
        self.glove_emb = Embedding()

    def compute_individual_metrics(self, ref, hyp):
        assert isinstance(hyp, six.string_types)
        ref = [a.strip() for a in ref]
        refs = {0: ref}
        ref_list = [ref]

        hyps = {0: [hyp.strip()]}
        hyp_list = [hyp]

        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                else:
                    ret_scores[method] = score

        if not self.no_glove:
            glove_hyps = [h.strip() for h in hyp_list]
            ref_list_T = self.np.array(ref_list).T.tolist()
            glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
            scores, scores_list_dict = self.eval_emb_metrics(glove_hyps, glove_refs, emb=self.glove_emb,
                                           metrics_to_omit=self.metrics_to_omit)
            scores = scores.split('\n')
            for score in scores:
                name, value = score.split(':')
                value = float(value.strip())
                ret_scores[name] = value

        return ret_scores

    def compute_metrics(self, ref_list, hyp_list):
        ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
        assert len(refs) == len(hyps)
        
        ret_score_list = {}
        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                        ret_score_list[m] = [float(each) for each in scs]
                else:
                    ret_scores[method] = score
                    ret_score_list[method] = [float(each) for each in scores]

        if not self.no_glove:
            glove_hyps = [h.strip() for h in hyp_list]
            ref_list_T = self.np.array(ref_list).T.tolist()
            glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
            scores, scores_list_dict = self.eval_emb_metrics(glove_hyps, glove_refs, emb=self.glove_emb)
            scores = scores.split('\n')
            for score in scores:
                name, value = score.split(':')
                value = float(value.strip())
                ret_scores[name] = value
            ret_score_list.update(scores_list_dict)
        
        return ret_scores, ret_score_list
