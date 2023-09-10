import re
import pandas as pd
import numpy as np
import itertools

from Levenshtein import distance as edit_distance
from Levenshtein import ratio as norm_edit_distance

# Rouge: Recall Oriented Understudy for Gisting Evaluation
# rouge scores for a reference/generated sentence pair
# source google seq2seq source code.
# transform recall to f1
class RougeEvaluator():
    def __init__(self, n_gram=2):
        self.n_gram = n_gram

    #supporting function
    def _split_into_words(self, sentences):
        """Splits multiple sentences into words and flattens the result"""
        return list(itertools.chain(*[re.sub(r'[()\[\]{}]', '', i).lower().strip().split() for i in sentences]))

    #supporting function
    def _get_ngrams(self, n, text):
        """Calcualtes n-grams.
            Args:
            n: which n-grams to calculate
            text: An array of tokens
            Returns:
            A set of n-grams
        """
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    #supporting function
    def _get_word_ngrams(self, n, sentences):
        """Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0

        words = self._split_into_words(sentences)
        return self._get_ngrams(n, words)

    def get_rouge_score(self, reference_sentences, evaluated_sentences):
        """
            Computes ROUGE-N of two text collections of sentences.
            Source: http://research.microsoft.com/en-us/um/people/cyl/download/
            papers/rouge-working-note-v1.3.1.pdf
            Args:
            evaluated_sentences: The sentences that have been picked by the summarizer (prediction)
            reference_sentences: The sentences from the referene set (truth)
            n: Size of ngram.  Defaults to 2.
            Returns:
            recall rouge score(float)
            Raises:
            ValueError: raises exception if a param has len <= 0
        """
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams = self._get_word_ngrams(self.n_gram, evaluated_sentences)
        reference_ngrams = self._get_word_ngrams(self.n_gram, reference_sentences)
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        # Handle edge case. This isn't mathematically correct, but it's good enough
        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-7))

        #just returning recall count in rouge, useful for our purpose
        return f1_score


class PostGenerationStageProcesser():
    def __init__(self,
        invalid_message="죄송합니다. 더 자세하게 물어봐 주시겠어요?",
        edit_distance_threshold=0.6,
        max_url_size=2,
        eval_rouge_n=[1, 2, 3],
        stopwords_path="./stopwords.txt",
        valid_urls_path="./valid_urls.txt",
    ):
        self.edit_distance_threshold = edit_distance_threshold
        self.max_url_size = max_url_size
        self.invalid_message = invalid_message
        self.evaluator = [RougeEvaluator(n) for n in eval_rouge_n]
        with open(stopwords_path, "r", encoding="utf8") as f:
            self.stopwords = [i for i in f.read().split("\n") if len(i.strip()) > 0]
        with open(valid_urls_path, "r", encoding="utf8") as f:
            self.valid_urls = [i for i in f.read().split("\n") if len(i.strip()) > 0]

    def pruner_extract_valid_output(self, rawtext):
        # extract answer part
        text = rawtext.split("### 답변:")
        # if can't check valid answer part, return false
        if len(text) < 2:
            return False
        else:
            lines = text[1]
            # if length of answer part is zero, return false
            if len(lines.strip()) == 0:
                return False
            else:
                output = []
                for i in lines.split("\n"):
                    tmp = " ".join(re.sub(r"[^가-힣a-zA-Z0-9.,'\/:()\[\]=&-_?<>@ ]", ' ', i).split())
                    if len(tmp) > 0:
                        output.append(tmp)
                return np.array(output)

    # pruning - sequence number duplicated
    def pruner_sequence_number_duplicated(self, lines):
        keep_lines = []
        searching_bucket = []
        for i in range(len(lines)):
            # continue if sentence is first location
            if i == 0:
                keep_lines.append(True)
            # if not, check stopping condition
            else:
                if lines[i][0].isdigit():
                    if lines[i][0] in searching_bucket:
                        keep_lines.append(False)
                    else:
                        keep_lines.append(True)
                else:
                    keep_lines.append(True)
            # save information on i th sentence
            if lines[i][0].isdigit():
                searching_bucket.append(lines[i][0])
        return np.array(keep_lines), "\n".join(lines[np.array(keep_lines)])

    # pruning - edit distance
    def pruner_norm_edit_distance(self, lines):
        keep_lines = []
        searching_bucket = []
        for i in range(len(lines)):
            # continue if sentence is first location
            if (i == 0) or ("https" in lines[i]) or (len(re.findall(r'^[0-9]', lines[i])) > 0):
                keep_lines.append(True)
            # if not, check stopping condition
            else:
                if (np.array([norm_edit_distance(lines[i], lines[j]) for j in range(i)]) > self.edit_distance_threshold).any():
                    keep_lines.append(False)
                else:
                    keep_lines.append(True)
        return np.array(keep_lines), "\n".join(lines[np.array(keep_lines)])

    # pruning - url number duplicated
    def pruner_limit_url_num_size(self, lines):
        keep_lines = []
        searching_bucket = 0
        for i in range(len(lines)):
            # continue if sentence is first location
            if i == 0:
                keep_lines.append(True)
            # if not, check stopping condition
            else:
                if "http" in lines[i]:
                    searching_bucket += 1
                    if searching_bucket > self.max_url_size:
                        keep_lines.append(False)
                    else:
                        keep_lines.append(True)
                else:
                    keep_lines.append(True)
        return np.array(keep_lines), "\n".join(lines[np.array(keep_lines)])

    # pruning - check sentence is complete 
    def pruner_complete_sentence(self, lines):
        keep_lines = []
        searching_bucket = 0
        for i in range(len(lines)):
            # continue if sentence is first location
            if (i == 0) or ("https" in lines[i]) or (len(re.findall(r'^[0-9]', lines[i])) > 0) or (len(re.findall(r'^[\[]', lines[i])) > 0):
                keep_lines.append(True)
            # if not, check stopping condition
            else:
                if len(re.findall(r'[.\n]$', lines[i])) < 1:
                    keep_lines.append(False)
                else:
                    keep_lines.append(True)
        return np.array(keep_lines), "\n".join(lines[np.array(keep_lines)])

    # pruning - check last sentence
    def pruner_last_sentence(self, lines):
        keep_lines = []
        searching_bucket = 0
        flag = True
        for i in range(len(lines)):
            # continue if sentence is first location
            if i == 0:
                keep_lines.append(True)
            # if not, check stopping condition
            else:
                if flag and (len(re.findall(r'^단,', lines[i])) > 0):
                    keep_lines.append(flag)
                    flag = False
                else:
                    keep_lines.append(flag)
        return np.array(keep_lines), "\n".join(lines[np.array(keep_lines)])

    # pruning - check stopword is in sentence
    def pruner_stopwords(self, lines):
        keep_lines = []
        searching_bucket = 0
        for i in range(len(lines)):
            # continue if sentence is first location
            if i == 0:
                keep_lines.append(True)
            # if not, check stopping condition
            else:
                flag = True
                for word in self.stopwords:
                    if word in lines[i]:
                        flag = False
                        break
                keep_lines.append(flag)
        return np.array(keep_lines), "\n".join(lines[np.array(keep_lines)])

    # pruning - check url is in valid urls
    def pruner_valid_urls(self, lines):
        keep_lines = []
        searching_bucket = 0
        for i in range(len(lines)):
            # continue if sentence is first location
            if i == 0:
                keep_lines.append(True)
            # if not, check stopping condition
            else:
                if "https" in lines[i]:
                    flag = False
                    keep_lines[-1] = False
                    for url in self.valid_urls:
                        if url in lines[i]:
                            flag = True
                            keep_lines[-1] = True
                            break
                    keep_lines.append(flag)
                else:
                    keep_lines.append(True)
        return np.array(keep_lines), "\n".join(lines[np.array(keep_lines)])

    def processing(self, text, reference_text=None):
        template = {"keep_lines": [], "prep_text": []}
        prun_result = {
            "pruner_sequence_number_duplicated": template.copy(),
            "pruner_norm_edit_distance": template.copy(),
            "pruner_limit_url_num_size": template.copy(),
            "pruner_complete_sentence": template.copy(),
            "pruner_last_sentence": template.copy(),
            "pruner_stopwords": template.copy(),
            "pruner_valid_urls": template.copy(),
        }

        # extract only valid character in string
        lines = self.pruner_extract_valid_output(text)

        # calculate rouge score before pruning process
        if reference_text is not None:
            eval_result = {
                "before_pruning": {"rouge_n1": 0.0, "rouge_n2": 0.0, "rouge_n3": 0.0, "rouge_nAvg": 0.0},
                "after_pruning": {"rouge_n1": 0.0, "rouge_n2": 0.0, "rouge_n3": 0.0, "rouge_nAvg": 0.0},
            }
            before_pruning_text = " ".join(lines)
            for idx, model in enumerate(self.evaluator):
                eval_result["before_pruning"][f"rouge_n{idx+1}"] = model.get_rouge_score(reference_text, before_pruning_text)
                eval_result["before_pruning"]["rouge_nAvg"] += eval_result["before_pruning"][f"rouge_n{idx+1}"] / len(self.evaluator)
        else:
            eval_result = False

        if isinstance(lines, tuple):
            return False, self.invalid_message
        else:
            output = self.pruner_sequence_number_duplicated(lines)
            prun_result["pruner_sequence_number_duplicated"]["keep_lines"] = output[0]
            prun_result["pruner_sequence_number_duplicated"]["prep_text"] = output[1]

            output = self.pruner_norm_edit_distance(lines)
            prun_result["pruner_norm_edit_distance"]["keep_lines"] = output[0]
            prun_result["pruner_norm_edit_distance"]["prep_text"] = output[1]

            output = self.pruner_limit_url_num_size(lines)
            prun_result["pruner_limit_url_num_size"]["keep_lines"] = output[0]
            prun_result["pruner_limit_url_num_size"]["prep_text"] = output[1]

            output = self.pruner_complete_sentence(lines)
            prun_result["pruner_complete_sentence"]["keep_lines"] = output[0]
            prun_result["pruner_complete_sentence"]["prep_text"] = output[1]

            output = self.pruner_last_sentence(lines)
            prun_result["pruner_last_sentence"]["keep_lines"] = output[0]
            prun_result["pruner_last_sentence"]["prep_text"] = output[1]

            output = self.pruner_stopwords(lines)
            prun_result["pruner_stopwords"]["keep_lines"] = output[0]
            prun_result["pruner_stopwords"]["prep_text"] = output[1]

            output = self.pruner_valid_urls(lines)
            prun_result["pruner_valid_urls"]["keep_lines"] = output[0]
            prun_result["pruner_valid_urls"]["prep_text"] = output[1]

            keep_lines = np.stack([v["keep_lines"] for v in prun_result.values()], axis=0).all(axis=0)
                
            # calculate rouge score after pruning process
            if reference_text is not None:
                after_pruning_text = " ".join(lines[keep_lines])
                for idx, model in enumerate(self.evaluator):
                    eval_result["after_pruning"][f"rouge_n{idx+1}"] = model.get_rouge_score(reference_text, after_pruning_text)
                    eval_result["after_pruning"]["rouge_nAvg"] += eval_result["after_pruning"][f"rouge_n{idx+1}"] / len(self.evaluator)
                
            return prun_result, eval_result, "\n".join(lines[keep_lines])
        

