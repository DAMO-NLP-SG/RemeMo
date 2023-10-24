from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
)
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import math



# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


@dataclass
class FlaxDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int
    num_add_special_tokens: int
    time_token_masking: bool
    time_mask_prob: float
    time_relation_prediction: bool
    time_cls_token_id: int
    scale_num_trelation_labels: float

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        if "cls_positions" in examples[0].keys():
            _cls_positions = [np.array(examples[i].pop("cls_positions")) for i in range(len(examples))]
        if "time_relation_labels" in examples[0].keys():
            _time_relation_labels = [np.array(examples[i].pop("time_relation_labels")) for i in range(len(examples))]
        if "time_tokens_mask" in examples[0].keys():
            _time_tokens_mask = [np.array(examples[i].pop("time_tokens_mask")) for i in range(len(examples))]

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])


        # if not self.time_token_masking:
        #     mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        # else: # Time Token Masking
        #     mask_indices = np.asarray([self.span_corruption_mask(expandend_input_length, _time_tokens_mask[i]) for i in range(batch_size)])
        ########################################
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        labels_with_padding_token_ids = self.filter_input_ids(input_ids, labels_sentinel)
        batch["labels"] = np.where(labels_with_padding_token_ids == self.pad_token_id, -100, labels_with_padding_token_ids)

        if self.time_token_masking and batch["input_ids"].shape[-1] == self.input_length + 1:
            batch["input_ids"] = batch["input_ids"][:, :-1]
        
        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )
        
        # Time Relation Prediction
        if self.time_relation_prediction:
            # remove those [TIME] tokens that are masked
            mask_indices = mask_indices.astype(np.int8)
            for batch_idx in range(len(_cls_positions)):
                try:
                    if (_time_relation_labels[batch_idx] == -100).all():
                        _time_relation_labels[batch_idx] = np.array([[-100]])
                        _cls_positions[batch_idx] = np.array([0])
                        continue
                    for cls_idx in range(len(_cls_positions[batch_idx])):
                        if _cls_positions[batch_idx][cls_idx] in np.where(mask_indices[batch_idx] == 1)[0]:
                            _cls_positions[batch_idx][cls_idx] = -1
                    truncated_cls_indices = np.where(_cls_positions[batch_idx] != -1)[0]
                    curr_time_relation_labels = []
                    for i in range(len(truncated_cls_indices)):
                        curr_time_relation_labels.append([_time_relation_labels[batch_idx][truncated_cls_indices[i], j] for j in truncated_cls_indices])
                    _time_relation_labels[batch_idx] = np.array(curr_time_relation_labels)
                    _cls_positions[batch_idx] = np.where(batch["input_ids"][batch_idx] == self.time_cls_token_id)[0]
                    assert _time_relation_labels[batch_idx].shape[0] == len(_cls_positions[batch_idx]),f"{_time_relation_labels[batch_idx].shape[0]} != {len(_cls_positions[batch_idx])}"
                    assert _time_relation_labels[batch_idx].shape[1] == len(_cls_positions[batch_idx]),f"{_time_relation_labels[batch_idx].shape[1]} != {len(_cls_positions[batch_idx])}"
                except Exception as e:
                    print(e)
                    # print()
                    # import pdb; pdb.set_trace()
                    _time_relation_labels[batch_idx] = np.array([[-100]])
                    _cls_positions[batch_idx] = np.array([0])
                    continue
            # pad cls_positions
            MAX_NUM_CLS = 10
            num_cls = [min(len(cls_position), MAX_NUM_CLS) for cls_position in _cls_positions]
            max_num_cls = max(num_cls)
            max_num_cls = max(max_num_cls, 1)
            # max_num_cls = min(max_num_cls, MAX_NUM_CLS)
            cls_positions = np.full([len(_cls_positions), max_num_cls], 0)
            for i in range(len(_cls_positions)):
                cls_positions[i, :min(len(_cls_positions[i]), MAX_NUM_CLS)] = _cls_positions[i][:min(len(_cls_positions[i]), MAX_NUM_CLS)]
            time_relation_labels = np.full([len(_time_relation_labels), max_num_cls, max_num_cls], -100)
            if self.scale_num_trelation_labels != 0.0:
                for i in range(len(_time_relation_labels)):
                    time_relation_labels[i, : min(_time_relation_labels[i].shape[0], MAX_NUM_CLS), : min(_time_relation_labels[i].shape[1], MAX_NUM_CLS)] \
                        = _time_relation_labels[i][: min(_time_relation_labels[i].shape[0], MAX_NUM_CLS), : min(_time_relation_labels[i].shape[1], MAX_NUM_CLS)]
            if self.scale_num_trelation_labels != -1.0 and self.scale_num_trelation_labels != 0.0:
                if self.scale_num_trelation_labels == 0.1:
                    print("##################")
                    print('Warning: "scale_num_trelation_labels = 0.1" means that there is only one TRC label in one instance. ')
                    print("##################")

                    mask = np.ones(time_relation_labels.shape, dtype=np.bool)
                    for i in range(len(time_relation_labels)):
                        for j in range(max_num_cls):
                            non_null_pairs = [(j,k) for k in range(max_num_cls) if time_relation_labels[i, j, k] != -100 and j != k]
                        if len(non_null_pairs) > 0:
                            chosen_pair = np.random.permutation(non_null_pairs)[0]
                            mask[i, chosen_pair[0], chosen_pair[1]] = False
                    time_relation_labels[mask] = -100
                else:
                    num_trelation_labels = int(min(self.scale_num_trelation_labels, max_num_cls) * max_num_cls * len(_time_relation_labels))
                    time_relation_labels = time_relation_labels.reshape(-1)
                    sample_indices = np.random.choice(np.arange(time_relation_labels.shape[0]), size=num_trelation_labels, replace=False)
                    mask = np.ones(time_relation_labels.shape, dtype=np.bool)
                    mask[sample_indices] = False
                    time_relation_labels[mask] = -100
                    time_relation_labels = time_relation_labels.reshape([len(_time_relation_labels), max_num_cls, max_num_cls])
            for i in range(len(_time_relation_labels)):
                for j in range(max_num_cls):
                    time_relation_labels[i, j, j] = -100

            batch["time_cls_indices"] = np.array(cls_positions)
            batch["time_relation_labels"] = np.array(time_relation_labels)
        ########################################

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            labels_with_padding_token_ids, self.pad_token_id, self.decoder_start_token_id
        )

        batch.pop("attention_mask", None)
        batch.pop("time_tokens_mask", None)
        batch.pop("special_tokens_mask", None)

        return batch.convert_to_tensors(tensor_type="pt")

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """

        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - self.num_add_special_tokens - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def get_time_token_mask_prob(self, input_ids, time_tokens_mask, time_mask_prob=0.3, max_time_mask_ratio=0.1):
        """
        Copy from ../pretrain/data_collator.py

        Get 0/1 labels for masked tokens with time-tokens-mask.
        Our goal is:
            to keep the total ratio of masks to be {15\%};
            If time-tokens does not account for more than {max_time_mask_ratio}, just sample [MASK] by {time_mask_prob};
            If time-tokens account for more than {max_time_mask_ratio}, we restrict the ratio to be {max_time_mask_ratio}, 
                so that {0.15-max_time_mask_ratio} normal tokens can get masked;
            If the sequence is too short, at least {one} time-token gets masked.
        Return:
            final_time_mask_prob:
                probability to mask time-tokens
            final_mlm_prob:
                probability to mask normal tokens
        """
        if self.tokenizer.pad_token_id in input_ids:
            input_ids = input_ids[:input_ids.index(self.tokenizer.pad_token_id)]
        # input_length = len(input_ids) - self.tokenizer.num_special_tokens_to_add(pair=False)
        input_length = len(input_ids)
        # time_tokens_length = sum(time_tokens_mask)
        time_tokens_length = np.sum(time_tokens_mask)
        if time_tokens_length > 0:
            try:
                if time_tokens_length / input_length <= max_time_mask_ratio:
                    final_time_mask_prob = time_mask_prob
                else:
                    final_time_mask_prob = time_mask_prob * (max_time_mask_ratio / (time_tokens_length / input_length))
                if time_tokens_length*final_time_mask_prob < 1:
                    final_time_mask_prob = 1 / time_tokens_length
                if input_length > time_tokens_length:
                    final_mlm_prob = (self.mlm_probability*input_length - final_time_mask_prob*time_tokens_length) / (input_length - time_tokens_length)
                else:
                    final_mlm_prob = 0.15
                return max(min(final_time_mask_prob, 1.0), 0.0), max(min(final_mlm_prob, 1.0), 0.0)
            except:
                return 0.0, self.mlm_probability
        else:
            return 0.0, self.mlm_probability
    
    def span_corruption_mask(self, input_length, time_tokens_mask, max_time_mask_ratio=0.1):
        """
        Copy from https://github.com/joeljang/Pretraining_T5_custom_dataset/blob/master/pretrain.py#L398
        I use this function to generate ``time-tokens-mask'' & ``normal mask'' for the span corruption task.
        30% of time-tokens should be masked, masked-time-tokens should not be more than 10% of all tokens. The rest should be normal tokens.
        """
        if self.mean_noise_span_length != 3:
            raise NotImplementedError("Only mean_noise_span_length=3 is supported if time_token_masking=True. ")
        max_index = input_length
        mask = max_index * [0]
        # span_num = math.ceil(( max_index * self.noise_density ) / self.mean_noise_span_length )
        num_noise_tokens = int(np.round(input_length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), input_length - 1)
        span_num = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length
        
        noise_span_lengths = _random_segmentation(num_noise_tokens, span_num)

        exclude = set([0])
        num_time_tokens = sum(time_tokens_mask)
        time_tokens_indices = np.where(time_tokens_mask)[0]
        time_tokens_indices_set = set(time_tokens_indices)
        for _ in exclude:
            if _ in time_tokens_indices_set:
                time_tokens_indices_set.remove(_)
        time_mask_indices = []
        already_mask_final = False
        for i in range(span_num):
            curr_noise_span_length = noise_span_lengths[i]
            while True:
                if not already_mask_final:
                    # rand_num = max_index - curr_noise_span_length - 1
                    rand_num = max_index - curr_noise_span_length
                    already_mask_final = True
                elif len(time_tokens_indices_set) == 0 \
                    or len(time_mask_indices) / input_length >= max_time_mask_ratio \
                        or len(time_mask_indices) / num_time_tokens >= self.time_mask_prob:
                    rand_num = np.random.randint(low=0, high=max_index) #Getting random number for mask index
                else:
                    rand_num = np.random.choice(time_tokens_indices)
                    if rand_num not in time_tokens_indices_set:
                        continue
                    time_tokens_indices_set.remove(rand_num)

                if all([rand_num + k not in exclude for k in range(curr_noise_span_length)]) and rand_num + curr_noise_span_length < max_index:
                    span = [rand_num + k for k in range(curr_noise_span_length)]
                    need_exclude = []
                    if rand_num + curr_noise_span_length < max_index:
                        need_exclude.append(rand_num + curr_noise_span_length)
                    if rand_num - 1 >= 0:
                        need_exclude.append(rand_num - 1)
                    for s in span:
                        mask[s] = 1
                        if s in time_tokens_indices:
                            time_mask_indices.append(s)
                        need_exclude.append(s)
                    for _ in need_exclude:
                        exclude.add(_) #Adding to exclude list
                        if _ in time_tokens_indices_set:
                            time_tokens_indices_set.remove(_)
                    break
        # mask[-1] = 1 # not meaningful, just to reach label_length=114
        return np.array(mask, dtype=bool)