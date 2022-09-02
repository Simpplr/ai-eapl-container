import itertools
import logging
from typing import Optional, Dict, Union
import torch

try:
    from .nlp_glob import nlp_func_map
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import nlp_func_map
    from nlp_ops import nlp_ops_funcs


def sent_tokenize(sentence, nlp):
    doc = nlp(sentence)
    return [sent.string.strip() for sent in doc.sents]


from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class QGPipeline:
    """Poor man's QG pipeline"""

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            ans_model: PreTrainedModel,
            ans_tokenizer: PreTrainedTokenizer,
            qg_format: str,
            use_cuda: bool
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer

        self.qg_format = qg_format

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if self.ans_model is not self.model:
            self.ans_model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

    def __call__(self, inputs: str, nlp):
        sents, answers = self._extract_answers(inputs, spacy_obj=nlp)
        flat_answers = list(itertools.chain(*answers))

        if len(flat_answers) == 0:
            return []

        if self.qg_format == "prepend":
            qg_examples = self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers)
        else:
            qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)

        qg_inputs = [example['source_text'] for example in qg_examples]
        questions = self._generate_questions(qg_inputs)
        output = [{'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, questions)]
        return output

    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_length=32,
            num_beams=4,
        )

        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions

    def _extract_answers(self, context, spacy_obj):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context, spacy_obj)
        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.ans_model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_length=32,
        )

        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
        answers = [item.split('<sep>') for item in dec]
        answers = [i[:-1] for i in answers]

        return sents, answers

    def _tokenize(self,
                  inputs,
                  padding=True,
                  truncation=True,
                  add_special_tokens=True,
                  max_length=512
                  ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs

    def _prepare_inputs_for_ans_extraction(self, text, spacy_obj):
        sents = sent_tokenize(text, nlp=spacy_obj)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()

            if self.model_type == "t5":
                source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs

    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            sent = sents[i]
            for answer_text in answer:
                sents_copy = sents[:]

                answer_text = answer_text.strip()

                ans_start_idx = sent.index(answer_text)

                sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text):]}"
                sents_copy[i] = sent

                source_text = " ".join(sents_copy)
                source_text = f"generate question: {source_text}"
                if self.model_type == "t5":
                    source_text = source_text + " </s>"

                inputs.append({"answer": answer_text, "source_text": source_text})

        return inputs

    def _prepare_inputs_for_qg_from_answers_prepend(self, context, answers):
        flat_answers = list(itertools.chain(*answers))
        examples = []
        for answer in flat_answers:
            source_text = f"answer: {answer} context: {context}"
            if self.model_type == "t5":
                source_text = source_text + " </s>"

            examples.append({"answer": answer, "source_text": source_text})
        return examples


class MultiTaskQAQGPipeline(QGPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, inputs: Union[Dict, str]):
        if type(inputs) is str:
            # do qg
            return super().__call__(inputs)
        else:
            # do qa
            return self._extract_answer(inputs["question"], inputs["context"])

    def _prepare_inputs_for_qa(self, question, context):
        source_text = f"question: {question}  context: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        return source_text

    def _extract_answer(self, question, context):
        source_text = self._prepare_inputs_for_qa(question, context)
        inputs = self._tokenize([source_text], padding=False)

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_length=16,
        )

        answer = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        return answer


class E2EQGPipeline:
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            use_cuda: bool
    ):

        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, context: str, nlp, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        input_length = inputs["input_ids"].shape[-1]

        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}.
        #         You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions

    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"

        inputs = self._tokenize([source_text], padding=False)
        return inputs

    def _tokenize(
            self,
            inputs,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs


SUPPORTED_TASKS = {
    "question-generation": {
        "impl": QGPipeline,
        "default": {
            "model": "valhalla/t5-small-qg-hl",
            "ans_model": "valhalla/t5-small-qa-qg-hl",
        }
    },
    "multitask-qa-qg": {
        "impl": MultiTaskQAQGPipeline,
        "default": {
            "model": "valhalla/t5-small-qa-qg-hl",
        }
    },
    "e2e-qg": {
        "impl": E2EQGPipeline,
        "default": {
            "model": "valhalla/t5-small-e2e-qg",
        }
    }
}


def pipeline(
        task: str,
        model: Optional = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        qg_format: Optional[str] = "highlight",
        ans_model: Optional = None,
        ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        use_cuda: Optional[bool] = True,
        **kwargs,
):
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)

    if task == "question-generation":
        if ans_model is None:
            # load default ans model
            ans_model = targeted_task["default"]["ans_model"]
            ans_tokenizer = AutoTokenizer.from_pretrained(ans_model)
            ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
        else:
            # Try to infer tokenizer from model or config name (if provided as str)
            if ans_tokenizer is None:
                if isinstance(ans_model, str):
                    ans_tokenizer = ans_model
                else:
                    # Impossible to guest what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )

            # Instantiate tokenizer if needed
            if isinstance(ans_tokenizer, (str, tuple)):
                if isinstance(ans_tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1])
                else:
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer)

            if isinstance(ans_model, str):
                ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)

    if task == "e2e-qg":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
    elif task == "question-generation":
        return task_class(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer,
                          qg_format=qg_format, use_cuda=use_cuda)
    else:
        return task_class(model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer,
                          qg_format=qg_format, use_cuda=use_cuda)


def eapl_nlp_t5_pipeline_init(data, cfg):
    t5_obj = cfg.get('t5_obj', 't5')
    model_name = cfg.get('model', 'valhalla/t5-base-e2e-qg')
    pipeline_name = cfg.get('pipeline', 'e2e-qg')
    model = pipeline(pipeline_name, model=model_name)
    data[t5_obj] = model

    return data


def eapl_t5_question_gen(nlp, text, spacy_nlp):
    return nlp(text, spacy_nlp)


def create_t5_question(data, text_dict, op_dict):
    t5 = op_dict.get('t5_obj', 't5')
    txt = op_dict.get('input_key', 'txt')
    questions = op_dict.get('out_key', 'questions')
    spacy_obj_key = op_dict.get('out_key', 'nlp')
    nlp = data[t5]
    text = text_dict[txt]
    spacy_obj = data[spacy_obj_key]
    output_tmp = eapl_t5_question_gen(nlp, text, spacy_nlp=spacy_obj)
    text_dict[questions] = list(set(output_tmp))
    return text_dict


t5_func_map = {
    'eapl_nlp_t5_pipeline_init': eapl_nlp_t5_pipeline_init,
    'create_t5_question': create_t5_question
}
nlp_func_map.update(t5_func_map)


def test_t5_pipeline():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_pipe', 'init_t5_pipe', 'record_nlp_ops'],
        'init_pipe': {
            'func': 'eapl_nlp_pipeline_init',
        },

        'init_t5_pipe': {
            'func': 'eapl_nlp_t5_pipeline_init',
            't5_obj': 't5'
        },

        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    'op': 'create_t5_question',
                    't5_obj': 't5',
                    'txt': 'txt',
                    'questions': 'questions_para'
                },
            ]
        }
    }

    data_test = {
        'text_obj': [
            {
                'txt': "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
            },
            {
                'txt': "Compatibility 2 of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
            }
        ]
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data_test = func(data_test, nlp_cfg)
    pprint(data_test['text_obj'])


if __name__ == '__main__':
    test_t5_pipeline()
