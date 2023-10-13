# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import typing as tp
from collections import OrderedDict, defaultdict, namedtuple
from logging import getLogger
from pathlib import Path

import fastBPE
import numpy as np
import torch

import codegen_sources.dataloaders.transforms as transf
from ..data.loader import DATASET_SPLITS
from ..trainer import get_programming_language_name
from ..utils import (
    add_noise,
    convert_to_text,
    REPO_ROOT,
)
import sys
sys.path.append(str(REPO_ROOT))

PathLike = tp.Union[Path, str]

SRC_ST_LANGS = "java"

TARGET_ST_LANG = {"cpp", "python"}

EVAL_OBF_PROBAS: tp.List[float] = []

BLEU_SCRIPT_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "multi-bleu.perl"
)
EVAL_DATASET_SPLITS = [ds for ds in DATASET_SPLITS if ds != "train"]
assert os.path.isfile(BLEU_SCRIPT_PATH)
ROOT_FOLDER = Path(__file__).parents[4]

logger = getLogger()
from .evaluator import (EncDecEvaluator,
                        gather_model_outputs)
from .gs_connector import GSConnector


class GSEvaluator(EncDecEvaluator):
    def __init__(self, trainer, data, params) -> None:
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.gs_model = trainer.gs_model
        self.params = params
        self.code_to_token_pipe = self.get_code_to_token_pipe()
        self.gs_connector = GSConnector()

    def eval_mode(self):
        pass

    def evaluate_mt(
        self,
        scores,
        data_set: str,
        lang1: str,
        lang2: str,
        eval_bleu: bool,
        eval_computation: bool,
        eval_computation_pivot: bool,
        eval_subtoken_score,
        span,
        deobfuscate=False,
        deobfuscate_probas=None,
        eval_ir_similarity=False,
    ):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        if data_set != "test":
            # skip
            return
        assert data_set in EVAL_DATASET_SPLITS
        assert lang1 in params.langs
        assert lang2 in params.langs
        rng = np.random.RandomState(0)
        torch_rng = torch.Generator().manual_seed(0)

        do_eval = {
            "bleu": eval_bleu,
            "st": params.eval_st,
            "computation": eval_computation,
            "computation_pivot": eval_computation_pivot,
            "subtoken_score": eval_subtoken_score,
            "ir_similarity": eval_ir_similarity,
        }

        bpe_model = None
        if do_eval["computation_pivot"]:
            bpe_model = fastBPE.fastBPE(params.pivot_bpe_model)  # type: ignore
            logger.info(f"Computing pivot CA for {lang1} to {lang2}")

        # store hypothesis to compute BLEU score
        if params.eval_bleu_test_only:
            datasets_for_bleu = ["test"]
        else:
            datasets_for_bleu = [s for s in EVAL_DATASET_SPLITS if s != "train"]

        lang2_id = params.lang2id[lang2]

        self.eval_mode()
        # encoder = self.encoder[0].module if params.multi_gpu else self.encoder[0]
        # decoder = (
        #     self.decoder[lang2_id] if params.separate_decoders else self.decoder[0]
        # )
        # decoder = decoder.module if params.multi_gpu else decoder
        encoder = None
        decoder = None

        for deobfuscation_proba in (
            deobfuscate_probas if deobfuscate_probas is not None else [None]
        ):
            if deobfuscate:
                rng = np.random.RandomState(0)

            word_metrics: tp.Mapping[str, float] = defaultdict(float)
            text_files: tp.Mapping[str, tp.Any] = defaultdict(list)

            logger.info(
                f"{params.global_rank}: generating MT hypotheses {lang1} -> {lang2}"
            )
            time_start_generate = time.perf_counter()
            will_compute_bleu = (
                any(
                    do_eval[k]
                    for k in (
                        "bleu",
                        "computation",
                        "subtoken_score",
                        "computation_pivot",
                    )
                )
                and data_set in datasets_for_bleu
            )
            for i, batch in enumerate(
                self.get_iterator(
                    data_set, lang1, lang2 if lang2 != lang1 else None, span=span
                )
            ):
                print(f"----------------------- iter {i=}")
                if i % params.world_size != params.global_rank:
                    continue  # Distribute batches on all GPUs

                computed_irs_upd, ir_creation_errors = [], None
                show_example = i == 0
                seq1, seq2, spans = self.extract_batch(
                    lang1,
                    lang2,
                    batch,
                    rng,
                    torch_rng,
                    deobfuscate,
                    deobfuscation_proba,
                    params,
                    do_eval["computation_pivot"],
                )
                if seq1 is None:
                    continue
                if do_eval["computation_pivot"]:
                    seq1, computed_irs_upd, ir_creation_errors = self.sequence_to_ir(
                        seq1, lang1, params, bpe_model
                    )
                text_files["computed_irs"].append(computed_irs_upd)
                enc1, dec2 = self.do_forward(
                    encoder, decoder, seq1, seq2, spans, params.fp16
                )
                # merge enc1 and dec2 (py and cpp code) to give to gs eval later
                enc1 = (seq1, seq2)

                self.update_word_metrics(
                    word_metrics,
                    seq2,
                    decoder,
                    dec2,
                    do_eval["computation_pivot"],
                    ir_creation_errors,
                )
                self.update_text_files(
                    text_files,
                    decoder,
                    seq1,
                    seq2,
                    enc1,
                    params,
                    lang1,
                    lang2,
                    data_set,
                    will_compute_bleu,
                    do_eval["computation_pivot"],
                    ir_creation_errors,
                    show_example,
                )

            time_hyp_generated = time.perf_counter()
            logger.info(
                f"Timing: Generated hypotheses in {time_hyp_generated - time_start_generate:.2f}s"
            )

            model_outputs = {**word_metrics, **text_files}
            if params.world_size > 1:
                torch.distributed.barrier()
                model_outputs_list = [None for _ in range(params.world_size)]
                torch.distributed.all_gather_object(model_outputs_list, model_outputs)
            else:
                model_outputs_list = [model_outputs]  # type: ignore

            if not params.is_master:
                continue

            model_outputs = gather_model_outputs(model_outputs_list)
            self.compute_metrics(
                model_outputs,
                data_set,
                lang1,
                lang2,
                params,
                scores,
                deobfuscate,
                deobfuscation_proba,
                do_eval,
                datasets_for_bleu,
                will_compute_bleu,
            )
            logger.info(
                f"Timing: Computed metrics in {time.perf_counter() - time_hyp_generated:.2f}s"
            )

    def extract_batch(
        self,
        lang1,
        lang2,
        batch,
        rng,
        torch_rng,
        deobfuscate,
        deobfuscation_proba,
        params,
        eval_computation_pivot,
    ):
        spans = None
        assert len(batch) >= 2
        if len(batch) == 2:
            if lang1 == lang2:
                x2, len2 = batch
                x1, len1 = add_noise(
                    x2, len2, self.params, len(self.data["dico"]) - 1, rng, torch_rng,
                )
            else:
                (x1, len1, ids1, len_ids1), (x2, len2, ids2, len_ids2) = batch
                assert x1 is not None
                if deobfuscate:
                    (x1, len1, x2, len2) = self.trainer.deobfuscate_by_variable(
                        x1,
                        x2,
                        deobfuscation_proba,
                        params.tokenization_mode == "roberta",
                        rng,
                    )
                if x1 is None:
                    return None, None, None
        elif len(batch) == 4:
            assert lang1 == lang2
            if eval_computation_pivot:
                x1, len1, _, _ = batch
                x2, len2 = x1, len1
            else:
                x2, len2, _, _ = batch
                x1, len1 = add_noise(
                    x2, len2, self.params, len(self.data["dico"]) - 1, rng, torch_rng,
                )

        else:
            assert len(batch) == 3
            (
                (x1, len1, ids1, len_ids1),
                (x2, len2, ids2, len_ids2),
                (spans, len_spans, _, _),
            ) = batch

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        # cuda
        # x1, len1, langs1, x2, len2, langs2, spans = to_cuda(
        #     x1, len1, langs1, x2, len2, langs2, spans
        # )

        return (x1, len1, langs1), (x2, len2, langs2), spans

    def do_forward(self, encoder, decoder, seq1, seq2, spans, is_fp16):
        x1, len1, langs1 = seq1
        x2, len2, langs2 = seq2
        
        # r1 = self.tokens_to_code(x1, len1, "python", self.params)
        # print(r1[0])
        # assert (self.code_to_tokens(r1, "python")==x1).all().item()
        # r2 = self.tokens_to_code(x2, len2, "python", self.params)
        return x1, x2

    def get_code_to_token_pipe(self):
        BPE_path = "data/bpe/cpp-java-python/codes"
        bpe_transf = transf.FastBpe(code_path=Path(BPE_path).absolute())
        bpetensorizer = transf.BpeTensorizer()
        bpetensorizer.dico = self.dico  # TODO: hacky
        in_pipe = bpe_transf.pipe(bpetensorizer)
        return in_pipe

    def code_to_tokens(self, code, lang):
        assert len(code) == 1, f"Batch size should be 1, got {len(code)}"
        ret = transf.CodeTokenizer(lang).apply(code[0])
        ret = self.code_to_token_pipe.apply(ret)
        return torch.LongTensor(ret).unsqueeze(-1)


    def update_word_metrics(
        self,
        word_metrics,
        seq2,
        decoder,
        dec2,
        eval_computation_pivot,
        ir_creation_errors,
    ):
        x2, len2, _ = seq2

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = (
            alen[:, None] < len2[None] - 1
        )  # do not predict anything given the last target word
        if eval_computation_pivot:
            # dec2: (len, bs, dim)
            err_mask = torch.BoolTensor([not err for err in ir_creation_errors]).to(
                x2.device
            )
            dec2 = dec2[:, err_mask]
            pred_mask = pred_mask[:, err_mask]
            y = x2[1:, err_mask].masked_select(pred_mask[:-1])
        else:
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

        # loss
        # word_scores, loss = decoder(
        #     "predict", tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True
        # )

        word_metrics["n_words"] += y.size(0)
        word_metrics["xe_loss"] += 1#loss.item() * len(y)
        word_metrics["n_valid"] += 1#(
        #     (word_scores.max(1)[1] == y).sum().item() if y.size(0) else 0
        # )


    def generate_mt_hypotheses(self, enc1, len1, lang2_id, decoder, params):
        len_v = (10 * len1 + 10).clamp(max=params.max_len)
        if params.beam_size == 1:
            if params.number_samples > 1:
                assert params.eval_temperature is not None
                generated, lengths = decoder.generate(
                    enc1.repeat_interleave(params.number_samples, dim=0),
                    len1.repeat_interleave(params.number_samples, dim=0),
                    lang2_id,
                    max_len=len_v.repeat_interleave(params.number_samples, dim=0),
                    sample_temperature=params.eval_temperature,
                )
                generated = generated.T.reshape(
                    -1, params.number_samples, generated.shape[0]
                ).T
                lengths, _ = lengths.reshape(-1, params.number_samples).max(dim=1)
            else:
                # generated, lengths = decoder.generate(
                #     enc1, len1, lang2_id, max_len=len_v
                # )
                generated, lengths = self.gen_mt_GS(enc1, len1, lang2_id)
            # print(f'path 1: {generated.shape}')

        else:
            assert params.number_samples == 1
            generated, lengths, _ = decoder.generate_beam(
                enc1,
                len1,
                lang2_id,
                beam_size=params.beam_size,
                length_penalty=params.length_penalty,
                early_stopping=params.early_stopping,
                max_len=len_v,
            )
            # print(f'path 2: {generated.shape}')

        text_hyps = convert_to_text(
            generated, lengths, self.dico, params, generate_several_reps=True,
        )
        return text_hyps, generated
    
    def gen_mt_GS(self, x, len_x, lang2_id):
        py_enc, cpp_enc = x
        py_x, py_len, _ = py_enc
        cpp_x, cpp_len, _ = cpp_enc
        py_x = self.tokens_to_code(py_x, py_len, "python", self.params)
        cpp_x = self.tokens_to_code(cpp_x, cpp_len, "cpp", self.params)
        print(f"{py_x=}")
        print(f"{cpp_x=}")
        cpp_function_def = cpp_x[0].split("{")[0]
        print(f"{cpp_function_def=}")
#         ret = ['bool isMajority ( int a [ ] , int n ) { unordered_map < int , int > mp ;\
# for ( int j = 0 ; j < n ; j ++ ) mp [ a [ j ] ] ++ ;\
# for ( auto x : mp ) if ( x.second \
# >= n / 2 ) return true ; return false ; }']
#         ret = ['int sumDigits ( int no ) {\n  return no == 0 ? 0 : no % 10 + sumDigits ( no / 10 ) ;\n}\n']
        # ret = ['int findSum ( int n ) {\n  int ans = 0 ;\n  for ( int i = 1 ;\n  i <= n ;\n  i ++ ) for ( int j = 1 ;\n  j <= n ;\n  j ++ ) ans += ( i / j ) ;\n  return ans ;\n}\n']
        print("-"*100)
        print(py_x[0])
        print(f"{len_x=}{lang2_id=}")
        py_x[0] = f"import numpy as np \nimport math\nfrom math import *\nimport collections\nfrom collections import *\nimport heapq\nimport itertools\nimport random\nimport sys\n\n{py_x[0]}"

        source_code_comments = "# Function definition:\n"
        source_code_comments += "# " + cpp_function_def + "\n"
        source_code_comments += "# Only generate tests for these types!\n"
        py_x[0] = source_code_comments + py_x[0]

        print("sending to gs:")
        print(py_x)

        ret = self.gs_connector.send_convert_request(py_x[0])
        print(f"{ret=}")
        ret = self.code_to_tokens(ret, "cpp")
        len_ret = torch.tensor([ret.shape[0]])
        return ret, len_ret
