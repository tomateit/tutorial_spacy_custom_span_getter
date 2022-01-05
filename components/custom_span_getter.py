from typing import Iterable, Callable, List
from functools import partial as p

import numpy as np
import spacy
import transformers
import spacy_transformers
import spacy_alignments as tokenizations
from transformers import AutoTokenizer
from spacy.tokens import Span, Doc



@spacy.registry.span_getters("transformer_aware_strided_spans.v1")
def transformer_aware_strided_spans_configurator(tokenizer_name: str, window: int, stride: int) -> Callable:
    """
    This components allows more dense chunking, than one can be achieved via word-wise chunking.
    """
    tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenizer = p(tokenizer_, padding=False, truncation=False)

    def transformer_aware_strided_spans(docs: Iterable[Doc]) -> List[List[Span]]:
        """
        Span getter shall, given a list of docs, produce for each doc a list of spans 
        to be feeded into a transformer model with limited window
        """
        tokenized_texts = tokenizer([doc.text for doc in docs])
        sentencepieces = map(tokenizer_.convert_ids_to_tokens, tokenized_texts.input_ids)
        list_of_lists_of_spans = []
 
        for doc, sentencepiece in zip(docs, sentencepieces):
            # get alignment between spacy tokenization and sentencpiece tokenization
            _, b2a = tokenizations.get_alignments([token.text for token in doc], sentencepiece)
            # Special tokens of sentencepiece tokenization have no match in original doc
            # thus have zero length, nevertheless affect length calculations
            # so we filter them out
            b2a = [t for t in b2a if len(t)] 
            
            # TRIVIAL CASE: a sentencepieced doc fits a window completely, so no need to do anything
            if len(sentencepiece) < window:
                # docs too short are pushed as whole
                list_of_lists_of_spans.append([doc[:]])
                continue

            buffer = []
            # Otherwise we iterate though sentencespices with given stride
            for i in range(0, len(sentencepiece), stride):
                sentencepice_window = b2a[i:i+window] # we always get the desired abount of tokens or less...
                if not len(sentencepice_window):
                    continue
                # But: each partially included word will at the end splitted as it was whole
                # so, to have a little ease with that
                # drop last word at all if it's not whole
                last_word = sentencepice_window[-1]
                current = sentencepice_window.count(last_word) # how many times word appears in current window
                total = sentencepiece.count(last_word) # how many times word appears in the whole tokenized sequence
                if total != current:
                    # in case a word was splitted in half we eliminate it from the end of current winodw
                    # and It will appear at the beginning of the next window
                    # If we do not do that, the resulting spans will be like:
                    # ["Mary had a little lamb", 
                    #  "lamb, Its fleece was white as snow"]
                    sentencepice_window = sentencepice_window[:-current]
                    # this will definitely cause the window shrinkage, which we will have to pad later
                    # but even this will be better, that straightforward word cutting

                # KEY POINT: I do not care about tokens anymore
                # I plainly infer from it a word span
                # This allows me to chose span window with flexibility, relatively to sentencepiece tokenization lengths
                sentencepice_window_flat = [index for sublist in sentencepice_window for index in sublist]
                if not len(sentencepice_window_flat):
                    continue
                min_ = min(sentencepice_window_flat)
                max_ = max(sentencepice_window_flat)
                span = doc[min_: max_ + 1]

                buffer.append(span)
            list_of_lists_of_spans.append(buffer)

        return list_of_lists_of_spans
    
    return transformer_aware_strided_spans