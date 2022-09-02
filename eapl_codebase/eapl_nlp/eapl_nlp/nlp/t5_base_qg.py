import numpy as np
import torch
import re
import random
import json
import pandas as pd
import logging

try:
    from .nlp_glob import nlp_func_map, nlp_glob
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import nlp_func_map, nlp_glob
    from nlp_ops import nlp_ops_funcs

t5_base_logger = logging.getLogger(__name__)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)


class QuestionGenerator:
    def __init__(self, model_dir=None):

        QG_PRETRAINED = 'iarfmoose/t5-base-question-generator'
        self.ANSWER_TOKEN = '<answer>'
        self.CONTEXT_TOKEN = '<context>'
        self.SEQ_LENGTH = 512

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED, use_fast=False)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)

        self.qa_evaluator = QAEvaluator(model_dir)

    def generate(self, article, spacy_obj, use_evaluator=True, num_questions=10, answer_style='all',
                 score_threshold=-4.0):

        t5_base_logger.debug("Generating questions...\n")

        qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style, spacy_obj)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        message = f"{len(generated_questions)} questions doesn't match {len(qg_answers)} answers"
        assert len(generated_questions) == len(qg_answers), message

        if use_evaluator:
            t5_base_logger.debug("Evaluating QA pairs...")
            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(generated_questions, qg_answers)
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)
            qa_list = self._get_ranked_qa_pairs(generated_questions, qg_answers, scores, score_threshold, num_questions)
        else:
            t5_base_logger.debug("Skipping evaluation step.")
            qa_list = self._get_all_qa_pairs(generated_questions, qg_answers)

        return qa_list

    def generate_qg_inputs(self, text, answer_style, spacy_obj):
        VALID_ANSWER_STYLES = ['all', 'sentences', 'multiple_choice']

        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(f"Invalid answer style {answer_style}. Please choose from {VALID_ANSWER_STYLES}")

        inputs = []
        answers = []

        if answer_style == 'sentences' or answer_style == 'all':
            segments = self._split_into_segments(text)
            for segment in segments:
                sentences = self._split_text(segment)
                prepped_inputs, prepped_answers = self._prepare_qg_inputs(sentences, segment)
                inputs.extend(prepped_inputs)
                answers.extend(prepped_answers)

        if answer_style == 'multiple_choice' or answer_style == 'all':
            sentences = self._split_text(text)
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_MC(sentences, spacy_obj)
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers

    def generate_questions_from_inputs(self, qg_inputs):
        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        return generated_questions

    def _split_text(self, text):
        MAX_SENTENCE_LEN = 128

        sentences = re.findall('.*?[.!\?]', text)

        cut_sentences = []
        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split('[,;:)]', sentence))
        # temporary solution to remove useless post-quote sentence fragments
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences

        return list(set([s.strip(" ") for s in sentences]))

    def _split_into_segments(self, text):
        MAX_TOKENS = 490

        paragraphs = text.split('\n')
        tokenized_paragraphs = [self.qg_tokenizer(p)['input_ids'] for p in paragraphs if len(p) > 0]

        segments = []
        while len(tokenized_paragraphs) > 0:
            segment = []
            while len(segment) < MAX_TOKENS and len(tokenized_paragraphs) > 0:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)
        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]

    def _prepare_qg_inputs(self, sentences, text):
        inputs = []
        answers = []

        for sentence in sentences:
            qg_input = '{} {} {} {}'.format(
                self.ANSWER_TOKEN,
                sentence,
                self.CONTEXT_TOKEN,
                text
            )
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    def _prepare_qg_inputs_MC(self, sentences, spacy_obj):
        spacy_nlp = spacy_obj
        docs = list(spacy_nlp.pipe(sentences, disable=['parser']))
        inputs_from_text = []
        answers_from_text = []

        for i in range(len(sentences)):
            entities = docs[i].ents
            if entities:
                for entity in entities:
                    qg_input = f'{self.ANSWER_TOKEN} {entity} {self.CONTEXT_TOKEN} {sentences[i]}'
                    answers = self._get_MC_answers(entity, docs)
                    inputs_from_text.append(qg_input)
                    answers_from_text.append(answers)

        return inputs_from_text, answers_from_text

    def _get_MC_answers(self, correct_answer, docs):

        entities = []
        for doc in docs:
            entities.extend([{'text': e.text, 'label_': e.label_} for e in doc.ents])

        # remove duplicate elements
        entities_json = [json.dumps(kv) for kv in entities]
        pool = set(entities_json)
        num_choices = min(4, len(pool)) - 1  # -1 because we already have the correct answer

        # add the correct answer
        final_choices = []
        correct_label = correct_answer.label_
        final_choices.append({'answer': correct_answer.text, 'correct': True})
        pool.remove(json.dumps({'text': correct_answer.text, 'label_': correct_answer.label_}))

        # find answers with the same NER label
        matches = [e for e in pool if correct_label in e]

        # if we don't have enough then add some other random answers
        if len(matches) < num_choices:
            choices = matches
            pool = pool.difference(set(choices))
            choices.extend(random.sample(pool, num_choices - len(choices)))
        else:
            choices = random.sample(matches, num_choices)

        choices = [json.loads(s) for s in choices]
        for choice in choices:
            final_choices.append({'answer': choice['text'], 'correct': False})
        random.shuffle(final_choices)
        return final_choices

    def _generate_question(self, qg_input):
        self.qg_model.eval()
        encoded_input = self._encode_qg_input(qg_input)
        with torch.no_grad():
            output = self.qg_model.generate(input_ids=encoded_input["input_ids"])
        question = self.qg_tokenizer.decode(output[0], skip_special_tokens=True)
        return question

    def _encode_qg_input(self, qg_input):
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_ranked_qa_pairs(self, generated_questions, qg_answers, scores, score_threshold, num_questions=10):
        if num_questions > len(scores):
            num_questions = len(scores)
            t5_base_logger.debug(
                f"Was only able to generate {num_questions} questions. For more questions, please input a longer text.")

        qa_list = []
        for i in range(num_questions):
            index, score = scores[i]
            if score < score_threshold:
                break
            qa = self._make_dict(
                generated_questions[index].split('?')[0] + '?',
                qg_answers[index],
                score)
            qa_list.append(qa)
        return qa_list

    def _get_all_qa_pairs(self, generated_questions, qg_answers):
        qa_list = []
        for i in range(len(generated_questions)):
            qa = self._make_dict(
                generated_questions[i].split('?')[0] + '?',
                qg_answers[i])
            qa_list.append(qa)
        return qa_list

    def _make_dict(self, question, answer, score=0):
        qa = dict()
        qa['question'] = question
        qa['answer'] = answer
        qa['score'] = float(score)
        return qa


class QAEvaluator:
    def __init__(self, model_dir=None):

        QAE_PRETRAINED = 'iarfmoose/bert-base-cased-qa-evaluator'
        self.SEQ_LENGTH = 512

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
        self.qae_model = AutoModelForSequenceClassification.from_pretrained(QAE_PRETRAINED)
        self.qae_model.to(self.device)

    def encode_qa_pairs(self, questions, answers):
        encoded_pairs = []
        for i in range(len(questions)):
            encoded_qa = self._encode_qa(questions[i], answers[i])
            encoded_pairs.append(encoded_qa.to(self.device))
        return encoded_pairs

    def get_scores(self, encoded_qa_pairs):
        scores = {}
        self.qae_model.eval()
        with torch.no_grad():
            for i in range(len(encoded_qa_pairs)):
                scores[i] = self._evaluate_qa(encoded_qa_pairs[i])

        return [(k, v) for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def _encode_qa(self, question, answer):
        if type(answer) is list:
            for a in answer:
                if a['correct']:
                    correct_answer = a['answer']
        else:
            correct_answer = answer
        return self.qae_tokenizer(
            text=question,
            text_pair=correct_answer,
            padding="max_length",
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        )

    def _evaluate_qa(self, encoded_qa_pair):
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]


def print_qa(qa_list, show_answers=True):
    for i in range(len(qa_list)):
        space = " " * int(np.where(i < 9, 3, 4))  # wider space for 2 digit q nums

        print("{}) Q: {}".format(i + 1, qa_list[i]["question"]))

        answer = qa_list[i]["answer"]

        # print a list of multiple choice answers
        if type(answer) is list:

            if show_answers:
                print(
                    "{}A: 1.".format(space),
                    answer[0]["answer"],
                    np.where(answer[0]["correct"], "(correct)", ""),
                )
                for j in range(1, len(answer)):
                    print(
                        "{}{}.".format(space + "   ", j + 1),
                        answer[j]["answer"],
                        np.where(answer[j]["correct"] == True, "(correct)", ""),
                    )

            else:
                print("{}A: 1.".format(space), answer[0]["answer"])
                for j in range(1, len(answer)):
                    print("{}{}.".format(space + "   ", j + 1), answer[j]["answer"])
            print("")

        # print full sentence answers
        else:
            if show_answers:
                print("{}A:".format(space), answer, "\n")


def eapl_nlp_t5_base_pipeline_init(data, cfg):
    t5_obj = cfg.get('t5_obj', 't5')
    model = nlp_glob.get(t5_obj, None)
    if model is None:
        model = QuestionGenerator()
        nlp_glob[t5_obj] = model
    data[t5_obj] = model

    return data


def eapl_qg_expand(data, cfg):
    input_key = cfg.get("input_key", "text_obj")
    out_key = cfg.get("out_key", input_key)
    explode_key = cfg['explode_key']
    doc_id_key = cfg.get('doc_id_key', None)
    q_key = cfg.get('q_key', 'question')
    qid_key = cfg.get("qid_key", "question_id")
    q_fmt = cfg.get('q_fmt', None)
    drop_nan_key = cfg.get('drop_key', qid_key)
    qg_ld = data[input_key]

    if doc_id_key:
        for doc_dict in qg_ld:
            for idx, q_dict in enumerate(doc_dict[explode_key]):
                q_dict[qid_key] = f"{doc_dict[doc_id_key]}_{idx}"

    qg_ld_df = pd.DataFrame(qg_ld)
    qg_ld_df = qg_ld_df.explode(explode_key)
    qg_ld_df[explode_key] = np.where(qg_ld_df[explode_key].isna(), {q_key: np.nan}, qg_ld_df[explode_key])
    qg_ld_df = pd.concat([qg_ld_df.drop([explode_key], axis=1), qg_ld_df[explode_key].apply(pd.Series)], axis=1)
    if q_fmt == 'list':
        qg_ld_df[q_key] = qg_ld_df[q_key].apply(lambda q: [q] if isinstance(q, str) else [])

    if drop_nan_key:
        if drop_nan_key in qg_ld_df.columns:
            qg_ld_df.dropna(subset=[drop_nan_key], inplace=True)
    data[out_key] = qg_ld_df.to_dict('records')

    return data


def create_t5_base_question(data, text_dict, op_dict):
    t5 = op_dict.get('t5_obj', 't5')
    txt = op_dict.get('input_key', 'txt')
    questions = op_dict.get('out_key', 'questions')
    answer_style = op_dict.get('answer_style', 'all')
    num_questions = op_dict.get('num_questions', 10)
    use_evaluator = op_dict.get('use_evaluator', True)
    score_threshold = op_dict.get('score_threshold', -4.0)
    spacy_obj_key = op_dict.get("spacy_obj_key", "nlp")
    qas = op_dict.get("qas", None)
    nlp = data[t5]
    spacy_obj = data[spacy_obj_key]
    text = text_dict[txt]

    output_tmp = nlp.generate(text, spacy_obj=spacy_obj, use_evaluator=use_evaluator, num_questions=num_questions,
                              answer_style=answer_style,
                              score_threshold=score_threshold)
    tmp_df = pd.DataFrame(output_tmp)
    tmp_df = tmp_df.drop_duplicates(subset=['question'])

    if qas:
        text_dict[qas] = tmp_df.to_dict('records')

    text_dict[questions] = list(tmp_df['question']) if 'question' in tmp_df.columns else []

    return text_dict


t5_base_func_map = {
    'eapl_nlp_t5_base_pipeline_init': eapl_nlp_t5_base_pipeline_init,
    'create_t5_base_question': create_t5_base_question,
    'eapl_qg_expand': eapl_qg_expand
}
nlp_func_map.update(t5_base_func_map)


def test_t5_base_pipeline():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_pipe', 'init_t5_pipe', 'record_nlp_ops', 'eapl_qg_expand', 'manage_data_keys'],

        'init_pipe': {
            'func': 'eapl_nlp_pipeline_init',
        },
        'init_t5_pipe': {
            'func': 'eapl_nlp_t5_base_pipeline_init',
            't5_obj': 't5'
        },

        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    'op': 'create_t5_base_question',
                    't5_obj': 't5',
                    'input_key': 'txt',
                    'out_key': 'questions',
                    'answer_style': 'sentences',
                    'qas': 'questions_rich',
                    'score_threshold': 0.0,
                    'num_questions': 100
                },
                {
                    'op': 'manage_text_dict_keys',
                    'pop_keys': ['questions']
                }
            ]
        },
        'eapl_qg_expand': {
            'func': 'eapl_qg_expand',
            'input_key': 'text_obj',
            'explode_key': 'questions_rich',
            'doc_id_key': 'input_id',
            'qid_key': 'question_id',
            'q_fmt': 'list'
        },
        'manage_data_keys': {
            'func': 'manage_data_keys',
            'pop_keys': ['t5']
        }
    }

    data_test = {
        'text_obj': [
            {
                'input_id': 1002,
                'txt': """Welcome to this quick take Rapid Learning module. Today's topic, 6 Managerial Styles you need to lead effectively. Rachel thought of herself as an enlightened manager, the opposite of the arrogant command-and-control jerk she'd worked for in a previous job. She respected her people. She listened to them soliciting their ideas and involving them in decision-making. She was extremely comfortable with this Democratic management style, which she'd mastered. Who wouldn't want to work for a nice respectful boss like Rachel. Well, a lot of people actually. Especially, high potential employees who want to work for top-performing teams and advance their careers. In this quick take you will learn what research says about the most effective quality of a strong leader. Six different managerial styles that all great leaders master and when and how to apply these styles. Rachel isn't a bad leader. But she's a one-trick pony a leader who over relies on a single managerial style. The one she happens to be most comfortable with. Research shows that strong leaders are far more flexible and adaptable than Rachel. They recognize that no one managerial style is appropriate in all cases. Leadership is situational and effective leaders must have a repertoire of managerial styles that allow them to effectively cope with a wide range of leadership challenges. The word develop is key. Leaders aren't born with the full package. It takes work. A leader like Rachel can only improve if she gets out of her comfort zone and learns to deploy managerial styles that don't come naturally to her. Let's look at 6 key managerial Styles and examine situations where they're appropriate or inappropriate. Let's begin with a command-and-control style. Think of an army General. I know how to do this. So do exactly what I tell you. Command and control tends to get universally condemned but that's not really fair. In the right situation, it's the best option. Command and control works when leading rank beginners who needs strong direction and it's the only style that fits in a real crisis. If the building is on fire or your computer systems crash leaders need to confidently take charge and give orders. That said command and control is ineffective in most other situations. managers often over rely on it because it feeds the ego. It feels good to believe that you have the answers but it disempowers and demoralizes people particularly high potential employees who seek autonomy and want to figure things out for themselves. A very different style is the Democratic style where you give team members a big say in decision-making. That's the approach that Rachel has relied on in the past. It's most effective for planning. When you get input from those who will implement the strategy plans will be reality-based and more comprehensive and people will take greater responsibility for the results. But as Leaders like Rachel eventually Discover, it can have a downside. While planning should be democratic, execution often requires a more top-down style. Imagine that your team falls behind on a project with a mission-critical deadline. That's not a debatable issue. You need to step in and light a fire under people. Next is the relating style. Great managers build rapport with the people. When bosses know the names of their employees' children and ask about them, they're using their relating style. They make people feel safe and valued. Rachel no doubt excelled at rapport-building. There couldn't be a downside to being a relator could there? Yes, many managers, and Rachel was probably one of them, rely too heavily on the relating style and find it difficult to make tough decisions. They don't want to be unpopular. They put protecting their people ahead of the organization's goals which in the long run will lead to an underperforming team. The next style, goal setting, is about communicating your vision and goals. You might say here's what we need to accomplish, I want you to figure out how. That last thought is essential. You're making goal-setting a collaborative effort where people buy in and feel empowered to execute. But there are two situations where this collaborative approach could backfire. The 1st is when you're driving a major change initiative. People resist change. They'll often nod in agreement with a goal but don't adopt a new behaviors required to achieve it. When that happens, you'll need them or directive approach. You'll also need to be more directive if your people lack the skills to execute. In other words, they're willing but not able. As a leader committed to a goal, you need either to skill up your existing team or bring in new talent. Another style, the Hands-On style, is the twin sister of command-and-control. When overused, it disempowers employees and destroys morale but it is appropriate when managers need to intervene and just get it done. If an employee says "I've tried everything but can't solve this problem, show me how", you deploy the Hands-On style and model the way. But be sure you're using this approach as an opportunity to teach a skill Hands-On managers are often high performers and their natural tendency is to jump in and fix problems. They tend to take all the credit for successes, which can create tremendous resentment on a team. Finally, there's the coaching style. Great coaches ask questions that help people find their own answers. They offer advice and direction and they follow up to ensure people are succeeding. Coaching isn't a quick fix. It's about the long-term development of people or managers who over rely on the command and control and hands-on styles, both of which achieve short-term objectives. The coaching style is the one that most needs to be developed. Of course, there are other managerial Styles you can use but these six give you a basic set of tools for most situations. One last thought: Managerial styles are like muscles in the body. They develop when you use them and atrophy when you don't. Look for appropriate opportunities to apply these styles and when you're in a situation where you're not getting the results you want, take a minute and ask yourself. What a different style work better? Thanks for watching."""
            }
        ],

        "non_editable_fields": ["question_id"]
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data_test = func(data_test, nlp_cfg)
    pprint(data_test)
    json_object = json.dumps(data_test, indent=4)

    # Writing to sample.json
    with open("336_output.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    test_t5_base_pipeline()
