# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
import math

import torch
from torch.utils.data import Dataset
import numpy as np


from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import datetime
import datetimerange
from filelock import FileLock

import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from generate_date_text import covert_date_to_text, generate_random_date_inside_span
from nltk.tokenize import word_tokenize
from utils import TIME_SPECIAL_TOKENS

# NSP_MODE_DICT = {
#     "none": 0,
#     "nsp": 1,
#     "temporal": 2,
# }

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
# logging.basicConfig(level=logging.INFO)
# logger.info("INFO")
# logger.warning("WARN")


TIME_NSP_CLS_LABELS = {
                        "earlier": 0,
                        "later": 1,
                        "contemporary": 2,
                        "random": -100,
                    }

def date_format_transform(date_tuple):
    """
        date_tuple: (20220102, 20220302)
    """
    return (
        "{}-{}-{}".format(date_tuple[0][:4], date_tuple[0][4:6], date_tuple[0][6:]),
        "{}-{}-{}".format(date_tuple[1][:4], date_tuple[1][4:6], date_tuple[1][6:])
    )

def time_gap_to_label(time_gap, threshold=180):
    if time_gap < -1 * 2 * threshold:
        return 0
    elif time_gap > 2 * threshold:
        return 1
    elif abs(time_gap) <= threshold:
        return 2
    else:
        return -100

def time_scale_to_label(time_gap_a, time_gap_b):
    """
    To fairly compare the scale between two time-gaps, we take ln()
        y = math.exp(x)
        ------------
        | x  |  y  |
        ------------
        0.0:  1.0000
        0.5:  1.6487
        1.0:  2.7183
        1.5:  4.4817
        2.0:  7.3891
        2.5:  12.1825
        3.0:  20.0855
        3.5:  33.1155
        4.0:  54.5982
        4.5:  90.0171
        5.0:  148.4132
        5.5:  244.6919
        6.0:  403.4288
        6.5:  665.1416
        7.0:  1096.6332
        7.5:  1808.0424
        8.0:  2980.9580
        8.5:  4914.7688
        9.0:  8103.0839
        9.5:  13359.7268
    """
    # If either of the time-gaps >= math.exp(6), we take a smaller ln_granularity=0.5 for scale comparison.
    # Otherwise, we take a larger ln_granularity=1.0; thus, the ln_granularity for time-gaps smaller than 400 days would not be too small.

    if time_gap_a == 0 and time_gap_b == 0:
        return 2
    elif time_gap_a == 0:
        return 0
    elif time_gap_b == 0:
        return 1

    if math.log(abs(time_gap_a)) >= 6 or math.log(abs(time_gap_b)) >= 6:
        ln_granularity = 0.5
    else:
        ln_granularity = 1.0
    ln_scale_gap = math.log(abs(time_gap_a)) - math.log(abs(time_gap_b))
    if abs(ln_scale_gap) <= ln_granularity:
        # gap_a and gap_b are similar in scale
        return 2
    elif ln_scale_gap < -1 * ln_granularity:
        # gap_a is smaller than gap_b
        return 0
    elif ln_scale_gap > ln_granularity:
        # gap_a is larger than gap_b
        return 1
    else:
        return -100

t5_prefixes = {
    "pretrain": "span-infilling: ",
    "time_pretrain": "span-infilling and time-relation-prediction: ",
    "qa": "question-answering: ",
    "time_qa": "question-answering and time-relation-prediction: ",
}

INSTANCE_MAX_LEN = 128


def cal_range_overlap(range_a, range_b):
    """
    e.g.,
        range_a = [20210101, 20210201]
        range_b = [20210202, 20210203]
    """
    assert len(range_a) == 2 and len(range_b) == 2
    range_a = (int(range_a[0]), int(range_a[1]))
    range_b = (int(range_b[0]), int(range_b[1]))
    if range_a[1] <= range_b[0]:
        return 0
    elif range_a[0] >= range_b[1]:
        return 1
    else:
        return 2

def merge_date_ranges(list_date_range):
    if len(list_date_range) == 0:
        return None
    output_range = None
    for i in range(0, len(list_date_range)):
        curr_datetimerange = datetimerange.DateTimeRange(date_format_transform(list_date_range[i])[0], date_format_transform(list_date_range[i])[1])
        if curr_datetimerange.is_valid_timerange() == False:
            continue
        if output_range is None:
            output_range = curr_datetimerange
        else:
            output_range = output_range.encompass(curr_datetimerange)
    output_range.start_time_format = "%Y%m%d"
    output_range.end_time_format = "%Y%m%d"
    return (output_range.get_start_time_str(), output_range.get_end_time_str())


class TemporalPretrainDatasetBuilder(object):
    """
    modified on the basis of HuggingFace's TextDatasetForNextSentencePrediction
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        text_column_name="text",
        truncation=True,
        data_args=None,
    ):
        self.data_args = data_args
        overwrite_cache = data_args.overwrite_cache

        self.temporal_label_set = TIME_NSP_CLS_LABELS

        self.truncation = truncation

        if data_args.use_t5_prefix:
            file_insertion = ""
        else:
            file_insertion = "no_t5_prefix."

        directory, filename = os.path.split(file_path)
        hf_dataset_dir = os.path.join(
            directory,
            f"hf_dataset.{file_insertion}{tokenizer.__class__.__name__}_{len(tokenizer)}_{block_size}.{filename}.dir",
        )

        # different nsp_mode share the same halfway cached file
        halfway_file_path = os.path.join(
            directory,
            f"cached_halfway.{file_insertion}{tokenizer.__class__.__name__}_{len(tokenizer)}_{block_size}.{filename}.pkl",
        )

        if not os.path.isfile(file_path):
            if overwrite_cache or not os.path.isfile(halfway_file_path) or not os.path.isdir(hf_dataset_dir):
                raise ValueError(f"Input file path {file_path} not found")

        self.tokenizer = tokenizer

        if os.path.exists(hf_dataset_dir) and not overwrite_cache:
            start = time.time()
            logger.info(
                f"Start loading features from cached file {hf_dataset_dir}"
            )
            if os.path.exists(os.path.join(hf_dataset_dir, "part-0")):
                self.dataset_list = []
                for i in range(20):
                    curr_hf_dataset_dir = os.path.join(hf_dataset_dir, f"part-{i}")
                    if not os.path.exists(curr_hf_dataset_dir):
                        break
                    self.dataset_list.append(load_from_disk(curr_hf_dataset_dir))
                # self.dataset = datasets.concatenate_datasets(dataset_list)
            else:
                self.dataset_list = [load_from_disk(hf_dataset_dir)]
            self.length_dataset = sum([len(_) for _ in self.dataset_list])
            logger.info(
                f"Successfully loaded {self.length_dataset} features from cached file {hf_dataset_dir} [took %.3f s]", time.time() - start
            )
        else:
            logger.info(f"Creating features from dataset file {filename}")

            self.documents = []
            # self.dates = []
            # self.time_tokens_mask = []
            self.examples = {
                    "input_ids": [],
                    # "attention_mask": [],
                    "special_tokens_mask": [],
                    "time_tokens_mask": [],
                    "cls_positions": [],
                    "time_relation_labels": [],
                }
            self.num_skipped = 0  # If error occurs, skip this example.

            assert file_path.split(".")[-1] == 'json'

            with open(file_path, 'r', encoding='utf-8') as f:
                sample_line = json.loads(f.readline())

            # if "dates" in sample_line:  # have `date` information, then create temporal-relation labels
            if "dates" in sample_line:
                if self.data_args.use_t5_prefix:
                    prefix = t5_prefixes["time_pretrain"]
                else:
                    prefix = ""
                self.prefix_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(prefix)
                )
                max_seq_len = block_size - len(tokenizer.tokenize(prefix))
                assert TIME_SPECIAL_TOKENS["bos_token"] in tokenizer.vocab and TIME_SPECIAL_TOKENS["eos_token"] in tokenizer.vocab

                FileLock(halfway_file_path+".lock").acquire()
                if os.path.exists(halfway_file_path) and not overwrite_cache:
                    FileLock(halfway_file_path+".lock").release()
                    start = time.time()
                    logger.info(
                        f"Start loading features from temporal halfway cached file {halfway_file_path}"
                    )
                    with open(halfway_file_path, "rb") as handle:
                        self.documents = pickle.load(handle)
                    logger.info(
                        f"Loaded {len(self.documents)} features from halfway-cached file {halfway_file_path} [took %.3f s]", time.time() - start
                    )
                else:
                    logger.info(f"Creating temporal halfway-features from dataset file {filename}")
                    num_dummy_instance = 0
                    lines = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # lines = [json.loads(_) for _ in f.readlines() if _.strip()]
                        for i, line in enumerate(f.readlines()):
                            if line.strip() == "":
                                continue
                            curr_line = json.loads(line)
                            if len(curr_line["dates"]) == 0:
                                continue

                            # Insert random date to improve generalization with a probability = 1%.
                            # First, we choose either the previous instance or the next one, denoted by sample_line. 
                            # Suppose the time-gap of sample_line["dates"][0] is "time_gap"
                            # The random date is sampled within the range of [sample_line_start_time - time_gap, sample_line_end_time + time_gap].
                            # Besides, a random number of mask_tokens are added to the random date text. 
                            if random.random() < 0.01:
                                if random.random() < 0.5:
                                    sample_line = lines[-1]
                                else:
                                    sample_line = curr_line
                                sample_time = [datetime.date.fromisoformat(str(_)) for _ in date_format_transform(sample_line["dates"][0])]
                                time_gap = sample_time[1].toordinal() - sample_time[0].toordinal()
                                random_date_text, random_date_span = generate_random_date_inside_span(
                                    span_start=sample_time[0].toordinal()-time_gap,
                                    span_end=sample_time[1].toordinal()+time_gap,
                                )
                                random_tokens = [TIME_SPECIAL_TOKENS['bos_token']] + word_tokenize(random_date_text) + [TIME_SPECIAL_TOKENS['eos_token']]
                                lines.append({
                                    "id": f"dummy_{num_dummy_instance}",
                                    "tokens": random_tokens,
                                    "text": " ".join(random_tokens),
                                    # "time_positions": [[1,2]],
                                    "time_positions": [[0,0]],
                                    # "time_char_positions": [[len("[TIME] "), len("[TIME] {}".format(tokenizer.mask_token))]], # (start, end)
                                    "time_char_positions": [[0,0]], # (start, end)
                                    "dates": [random_date_span],
                                })
                                num_dummy_instance += 1
                            lines.append(curr_line)
                            #if i >= 40960:
                            #    break

                    # # load id-to-sent file for temporal-wiki
                    # with open(os.path.join(directory, "id2doc.json"), 'r', encoding='utf-8') as f:
                    #     id2doc = json.load(f)
                    
                    texts = [_["text"].strip() for _ in lines]

                    tokenized = dict()
                    if "bookcorpus" in file_path:
                        tokenize_batch_size = 100
                    else:
                        tokenize_batch_size = 1000
                    for batch_count in tqdm(range(-(-len(texts)//tokenize_batch_size)), desc=f"Tokenizing file {filename}"):
                        curr_batch = texts[batch_count*tokenize_batch_size : (batch_count+1)*tokenize_batch_size]
                        curr_tokenized = tokenizer(
                            curr_batch,
                            # max_length=max_seq_len,
                            # stride=0,
                            return_offsets_mapping=True,
                            # return_overflowing_tokens=True,
                            return_special_tokens_mask=True,
                            add_special_tokens=False,
                        )
                        for k, v in curr_tokenized.items():
                            if k not in tokenized:
                                tokenized[k] = []
                            tokenized[k].extend(v)

                    logger.info(f"Filtering out too long temporal-sentences that > {INSTANCE_MAX_LEN } (after tokenization) ...")
                    for i in range(len(tokenized["input_ids"])):
                        if len(tokenized["input_ids"][i]) > INSTANCE_MAX_LEN:
                            continue
                        curr_instance = dict()
                        for k, v in tokenized.items():
                            curr_instance[k] = v[i]
                        for k in ["id", "time_positions", "time_char_positions", "dates"]:
                            curr_instance[k] = lines[i][k]
                        curr_instance["text"] = texts[i]
                        self.documents.append(curr_instance)
                    del lines, tokenized, texts
                        
                    # create cache dump
                    start = time.time()
                    with open(halfway_file_path, "wb") as handle:
                        pickle.dump(self.documents, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        f"Saving features into halfway-cached file {halfway_file_path} [took {time.time() - start:.3f} s]"
                    )
                    FileLock(halfway_file_path+".lock").release()

                self.create_temporal_examples_from_document(block_size=block_size)

                avg_num_cls = sum([len(_) for _ in self.examples["cls_positions"]]) / len(self.examples["cls_positions"])
                logger.info("Average number of {} tokens per example: {}".format(TIME_SPECIAL_TOKENS["bos_token"], avg_num_cls))

                all_time_relation_labels = [0, 0, 0]
                for i in range(len(self.examples["time_relation_labels"])):
                    for j in range(len(self.examples["time_relation_labels"][i])):
                        for k in range(len(self.examples["time_relation_labels"][i][j])):
                            all_time_relation_labels[self.examples["time_relation_labels"][i][j][k]] += 1
                logger.info("Time relation labels: {}".format(all_time_relation_labels))
                sum_time_relation_labels = sum(all_time_relation_labels)
                logger.info("Time relation labels: {}".format([_ / sum_time_relation_labels for _ in all_time_relation_labels]))

            #### NOT Temporal ####
            else:
                if self.data_args.use_t5_prefix:
                    prefix = t5_prefixes["pretrain"]
                else:
                    prefix = ""
                self.prefix_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(prefix)
                )
                max_seq_len = block_size - len(self.prefix_ids)
                # with FileLock(halfway_file_path+".lock"):
                FileLock(halfway_file_path+".lock").acquire()
                if os.path.exists(halfway_file_path) and not overwrite_cache:
                    FileLock(halfway_file_path+".lock").release()
                    start = time.time()
                    logger.info(
                        f"Start loading features from non-temporal halfway cached file {halfway_file_path}"
                    )
                    with open(halfway_file_path, "rb") as handle:
                        self.documents = pickle.load(handle)
                    logger.info(
                        f"Loaded {len(self.documents)} features from halfway-cached file {halfway_file_path} [took %.3f s]", time.time() - start
                    )
                else:
                    logger.info(f"Creating non-temporal halfway-features from dataset file {filename}")
                    lines = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f.readlines():
                            if line.strip() == "":
                                continue
                            line = json.loads(line)
                            if text_column_name not in line:
                                continue
                            lines.append(line)
                    if "bookcorpus" in file_path:
                        _texts = ' '.join([' '.join(_[text_column_name]) for _ in lines])
                        texts = []
                        len_text = len(_texts)
                        num_shards = int(len_text / 1000000)
                        mini_len_text = int(len_text / num_shards)
                        for idx in range(num_shards):
                            if idx == num_shards - 1:
                                texts.append(_texts[idx*mini_len_text:])
                            else:
                                texts.append(_texts[idx*mini_len_text : (idx+1)*mini_len_text])
                        del lines, _texts
                    else:
                        texts = ' '.join([' '.join(_[text_column_name]) for _ in lines])
                        del lines
                    tokenized = {"input_ids": []}
                    if "bookcorpus" in file_path:
                        tokenize_batch_size = 1000
                    else:
                        tokenize_batch_size = 1000
                    for batch_count in tqdm(range(-(-len(texts)//tokenize_batch_size)), desc=f"Tokenizing file {filename}"):
                        curr_batch = texts[batch_count*tokenize_batch_size : (batch_count+1)*tokenize_batch_size]
                        curr_tokenized = tokenizer(
                            text=curr_batch,
                            max_length=max_seq_len,
                            stride=0,
                            return_offsets_mapping=True,
                            return_overflowing_tokens=True,
                            return_special_tokens_mask=True,
                        )
                        tokenized["input_ids"].extend(curr_tokenized['input_ids'])
                        # for k, v in curr_tokenized.items():
                        #     if k not in tokenized:
                        #         tokenized[k] = []
                        #     tokenized[k].extend(v)

                    for i in range(len(tokenized["input_ids"])):
                        self.documents.append(
                            self.prefix_ids + tokenized["input_ids"][i]
                        )
                    del tokenized
                    # create cache dump
                    start = time.time()
                    with open(halfway_file_path, "wb") as handle:
                        pickle.dump(self.documents, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        f"Saving features into halfway-cached file {halfway_file_path} [took {time.time() - start:.3f} s]"
                    )
                    FileLock(halfway_file_path+".lock").release()

                # for doc_index, document in enumerate(tqdm(self.documents, desc="Building NSP Labels")):
                #     self.create_examples_from_document(document, doc_index, block_size)
                self.create_examples_from_document(block_size=block_size)
            logger.info("Finished building sentence labels, with {} skipped segments. ".format(self.num_skipped))

            # import pdb; pdb.set_trace()

            # create cache dump
            start = time.time()
            # with open(hf_dataset_dir, "wb") as handle:
            #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(hf_dataset_dir, 'w', encoding='utf-8') as handle:
            #     json.dump(self.examples, handle)
            save_batch_size = len(self.examples["input_ids"]) // 3
            self.dataset_list = []
            for batch_count in tqdm(range(-(-len(self.examples["input_ids"])//save_batch_size)), desc=f"Saving {filename} dataset to disk in batches. "):
                curr_batch = dict()
                for k, v in self.examples.items():
                    curr_batch[k] = v[batch_count*save_batch_size : (batch_count+1)*save_batch_size]
                self.dataset_list.append(Dataset.from_dict(curr_batch))
                self.dataset_list[-1].save_to_disk(
                    os.path.join(hf_dataset_dir, f"part-{batch_count}")
                )
            # self.dataset = datasets.concatenate_datasets(dataset_list)
            self.length_dataset = sum([len(_) for _ in self.dataset_list])
            logger.info(
                f"Finshed saving {self.length_dataset} features into cached file {hf_dataset_dir} [took {time.time() - start:.3f} s]"
            )
            # FileLock(lock_path).release()
            self.documents = []
            self.examples = []
        random_index = random.randint(0, len(self.dataset_list[0]))
        logger.info("Random sample from part-0 of {}".format(filename))
        logger.info(self.dataset_list[0][random_index])

    def create_examples_from_document(self, block_size: int):
        """Create examples from non-temporal documents"""

        # max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.

        random.shuffle(self.documents)

        for input_ids in tqdm(self.documents, desc="Building non-temporal instances"):
            if len(input_ids) < block_size:
                difference = block_size - len(input_ids)
                _ = input_ids + [self.tokenizer.pad_token_id]*difference
            elif len(input_ids) > block_size:
                raise ValueError(f"Found len(input_ids) > block_size: {len(input_ids)} > {block_size}")
            else:
                _ = input_ids
            self.examples["input_ids"].append(np.array(_))
            # self.examples["attention_mask"].append(np.array([1]*len(_)))
            self.examples["special_tokens_mask"].append(np.array(self.tokenizer.get_special_tokens_mask(_, already_has_special_tokens=True)))
            self.examples["time_tokens_mask"].append(np.array([0]*len(_), dtype=np.int8))
            self.examples["cls_positions"].append(np.array([0], dtype=np.int16))
            self.examples["time_relation_labels"].append(np.array([[-100]], dtype=np.int8))

    def create_temporal_examples_from_document(self, block_size: int):
        """Create examples from temporal factual sentences"""

        num_skipped_because_offsets_mismatch = 0
        num_skipped_because_offsets_end_mismatch = 0
        num_all_time_positions = 0
        
        # Initialize
        current_chunk = []
        current_chunk.extend(self.prefix_ids)
        time_tokens_mask = [0]*len(self.prefix_ids)
        cls_positions = []
        date_ranges = []

        pbar = tqdm(total = len(self.documents), desc="Building temporal instances (sent-level)")
        doc_index = 0
        # for doc_index, document in enumerate(tqdm(self.documents, desc="Building temporal instances (sent-level). ")):
        while doc_index < len(self.documents):
            document = self.documents[doc_index]
            if len(current_chunk) == 0 and (len(document["input_ids"]) + len(self.prefix_ids) + 1) > block_size:
                pbar.update(1)
                doc_index += 1
                continue

            if doc_index == len(self.documents) - 1 or (len(current_chunk) + len(document["input_ids"]) + 1 > block_size):
                current_chunk.append(self.tokenizer.eos_token_id)
                time_tokens_mask.append(0)
                if len(current_chunk) < block_size:
                    difference = block_size - len(current_chunk)
                    current_chunk.extend([self.tokenizer.pad_token_id]*difference)
                    time_tokens_mask.extend([0]*difference)
                if len(cls_positions) == 0:
                    self.num_skipped += 1
                else:
                    time_relation_labels = np.full([len(cls_positions), len(cls_positions)], -100)
                    # cls_positions = [cls_positions[i] for i in range(len(date_ranges)) if date_ranges[i] != None]
                    # date_ranges = [_ for _ in date_ranges if _ != None]
                    for i in range(len(date_ranges)):
                        if date_ranges[i] == None:
                            continue
                        for j in range(len(date_ranges)):
                            if date_ranges[j] == None:
                                continue
                            time_relation_labels[i,j] = cal_range_overlap(date_ranges[i], date_ranges[j])
                    assert len(current_chunk) == len(time_tokens_mask)
                    
                    self.examples["input_ids"].append(current_chunk)
                    # self.examples["attention_mask"].append([1]*len(current_chunk))
                    self.examples["special_tokens_mask"].append(self.tokenizer.get_special_tokens_mask(current_chunk, already_has_special_tokens=True))
                    self.examples["time_tokens_mask"].append(np.array(time_tokens_mask, dtype=np.int8))
                    self.examples["cls_positions"].append(np.array(cls_positions, dtype=np.int16))
                    self.examples["time_relation_labels"].append(np.array(time_relation_labels, dtype=np.int8))

                # Re-Initialize
                current_chunk = []
                current_chunk.extend(self.prefix_ids)
                time_tokens_mask = [0]*len(self.prefix_ids)
                cls_positions = []
                date_ranges = []
                if doc_index == len(self.documents) - 1:
                    break

            offsets_starts = []
            offsets_ends = []
            for _ in document["offset_mapping"]:
                if _ is not None:
                    offsets_starts.append(_[0])
                    offsets_ends.append(_[1])
                else:
                    offsets_starts.append(_)
                    offsets_ends.append(_)

            cls_positions.append(len(current_chunk))
            current_chunk += document["input_ids"]
            date_ranges.append(merge_date_ranges(document["dates"]))

            curr_time_tokens_mask = [0]*len(document["input_ids"])

            for time_idx, time_char_position in enumerate(document['time_char_positions']):
                # Detect if the answer is out of the span. In such case, skip this [TIME].
                if not (document["offset_mapping"][0][0] <= time_char_position[0] and document["offset_mapping"][-1][1] >= time_char_position[1]):
                    raise IndexError("Time span is out of the sentence.")
                # Detect time-range == None
                if document["dates"][time_idx] == None:
                    continue
                if time_char_position[0] == time_char_position[1]:
                    continue
                num_all_time_positions += 1
                continue_signal = False
                try:
                    time_start_position = offsets_starts.index(time_char_position[0])
                except Exception as e:
                    # num_skipped_because_offsets_mismatch += 1
                    # continue_signal = True
                    try:
                        time_start_position = offsets_starts.index(time_char_position[0]+1)
                    except:
                        try:
                            time_start_position = offsets_starts.index(time_char_position[0]-1)
                        except:
                            num_skipped_because_offsets_mismatch += 1
                            continue_signal = True
                try:
                    time_end_position = offsets_ends.index(time_char_position[1])
                except Exception as e:
                    # num_skipped_because_offsets_end_mismatch += 1
                    # continue_signal = True
                    try:
                        time_end_position = offsets_ends.index(time_char_position[1]+1)
                    except:
                        try:
                            time_end_position = offsets_ends.index(time_char_position[1]-1)
                        except:
                            num_skipped_because_offsets_end_mismatch += 1
                            continue_signal = True
                    print("#"*100)
                    print("input_ids: {}".format(document["input_ids"]))
                    print("offsets: {}".format(document["offset_mapping"]))
                    print("time_char_position: {}".format(time_char_position))
                    raise e
                if continue_signal:
                    continue
                curr_time_tokens_mask[time_start_position:time_end_position+1] = [1]*(time_end_position + 1 - time_start_position)
            time_tokens_mask += curr_time_tokens_mask
            
            pbar.update(1)
            # pbar.set_postfix_str("start_mismatch {} / {} = {:.2f}; end_mismatch: {} / {} = {:.2f}".format(num_skipped_because_offsets_mismatch, num_all_time_positions, num_skipped_because_offsets_mismatch/num_all_time_positions, num_skipped_because_offsets_end_mismatch, num_all_time_positions, num_skipped_because_offsets_end_mismatch/num_all_time_positions))
            doc_index += 1
        pbar.close()
        # self.examples["num_skipped_because_offsets_mismatch"] = num_skipped_because_offsets_mismatch
        # self.examples["num_all_time_positions"] = num_all_time_positions
        logger.info("[Time-entity offsets mismatch (skipped) ] start_mismatch: {} / {} = {:.2f}; end_mismatch: {} / {} = {:.2f}".format(num_skipped_because_offsets_mismatch, num_all_time_positions, num_skipped_because_offsets_mismatch/num_all_time_positions, num_skipped_because_offsets_end_mismatch, num_all_time_positions, num_skipped_because_offsets_end_mismatch/num_all_time_positions))
        # logger.info(f"Skipped {num_skipped_because_offsets_mismatch} time-tokens-mask out of total {num_all_time_positions} time-entities, because offsets mismatch with time_char_positions. ")
