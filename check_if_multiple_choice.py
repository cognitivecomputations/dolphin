# From https://github.com/google-research/FLAN/blob/main/flan/templates.py
# Modified to be used for figuring out which one is multiple choice


def rte_check(string):
    return not string.startswith("Generate a context and a hypothesis")


def cosmos_qa_check(string: str):
    if string.startswith("Write a question about the article"):
        return False
    elif string.endswith("Generate a question about the above context."):
        return False
    else:
        return True


def ag_news_subset_check(string: str):
    [
        ("{title}\n\n{text}\n\nWhat is this text about?\n{options_}", "{answer}"),
        ("{title}\n\n{text}\n\nWhich topic is this article about?\n{options_}", "{answer}"),
        ("{text}\nWhich is the best summary of this article?\n{options_}", "{answer}"),
        ("{text}\nWhat is this text about?\n{options_}", "{answer}"),
        ("{text}\n\nWhat best summarizes the content of the above article?\n{options_}", "{answer}"),
        ("Which is this about?\n\n{text}\n\n{options_}", "{answer}"),
        ("Which is an appropriate title for this article?\n\n{text}\n\n{options_}", "{answer}"),
        ("Select the topic that this about:\n\n{text}\n\n{options_}", "{answer}"),
        ("Write a title:\n{text}", "{title}"),
        ("{text}\n\nWhat is a good title for this?", "{title}"),
    ]
    if string.startswith("Write a title:"):
        return False
    elif string.endswith("What is a good title for this?"):
        return False
    else:
        return True


def imdb_reviews_check(string: str):
    if string.startswith("Write a"):
        return False
    elif string.startswith("Generate a movie review with"):
        return False
    elif string.startswith("What's an example of a movie review?"):
        return False
    else:
        return True


def paws_wiki_check(string: str):
    if string.startswith("Please check if these have the same meaning. Answer \"yes\" if they do, otherwise \"no\"."):
        return False
    else:
        return True


def sentiment140_check(string: str):
    if string.startswith("Generate a tweet that has the following sentiment: "):
        return False
    elif string.startswith("Write a "):
        return False
    elif string.startswith("What is an example of a tweet?"):
        return False
    else:
        return True


def story_cloze_check(string: str):
    if string.startswith("Write a story that ends with this"):
        return False
    elif string.startswith("Write a plausible story that ends with this sentence?"):
        return False
    else:
        return True


def copa_check(string: str):
    if string.startswith("Write a sentence."):
        return False
    elif string.startswith("Write two sentences."):
        return False
    else:
        return True


def yelp_polarity_reviews_check(string: str):
    if string.startswith("What would be an example of an "):
        return False
    elif string.startswith("Generate a "):
        return False
    elif string.startswith("Write a "):
        return False
    else:
        return True


def arc_check(string: str):
    if string.startswith("Write a question you would see in a school textbook."):
        return False
    elif string.startswith("What's an example of a grad-school level question?"):
        return False
    elif string.startswith("I just took a test in school today. What question was I asked?"):
        return False
    else:
        return True


def anli_check(string: str):
    if string.startswith("Generate a context and a hypothesis."):
        return False
    else:
        return True


def multirc_check(string: str):
    if string.endswith("Do you have any questions?"):
        return False
    elif string.endswith("What question would one ask from this paragraph?"):
        return False
    else:
        return True


def cb_check(string: str):
    if string.startswith("Generate a context and a hypothesis."):
        return False
    else:
        return True


def cola_check(string: str):
    if string.startswith("Generate short a sentence that is linguistically"):
        return False
    elif string.startswith("Produce a brief English sentence that would be considered grammatically"):
        return False
    else:
        return True


def sst2_check(string: str):
    if string.startswith("Write a "):
        return False
    elif string.startswith("Generate a short movie review that has"):
        return False
    else:
        return True


def qnli_check(string: str):
    if string.startswith("Can you generate a question with a factual answer?"):
        return False
    else:
        return True


def snli_check(string: str):
    if string.startswith("Write a brief sentence."):
        return False
    else:
        return True


def trec_check(string: str):
    if string.startswith("Please ask me a question."):
        return False
    else:
        return True


def stsb_check(string: str):
    if string.endswith(
            "Generate a new sentence that is, on a scale from 0 to 5, a {answer_str} in textual similarity to the above sentence."):
        return False
    elif string.endswith("out of 5 in terms of textual similarity to the above sentence?"):
        return False
    else:
        return True


def piqa_check(string: str):
    if string.startswith(
            "What's an example of a task that requires knowledge of physical objects to perform?"):
        return False
    elif string.startswith("What kind of task would test someone's ability to perform physical reasoning?"):
        return False
    else:
        return True


def openbookqa_check(string: str):
    if string.startswith(
            "What sentence would provide a factual answer to this question:"):
        return False
    elif string.startswith("What is a random fact?"):
        return False
    elif string.startswith("Generate a sentence that contains a fact."):
        return False
    else:
        return True


PATTERNS = {
    "rte": rte_check,
    "wsc": lambda x: True,
    "wsc273": lambda x: True,
    "wic": lambda x: True,
    "record": lambda x: True,
    "natural_questions": lambda x: False,
    "trivia_qa": lambda x: False,
    "math_dataset": lambda x: False,
    "aeslc": lambda x: False,
    "cnn_dailymail": lambda x: False,
    "gigaword": lambda x: False,
    "multi_news": lambda x: False,
    "newsroom": lambda x: False,
    "samsum": lambda x: False,
    "xsum": lambda x: False,
    "squad_v1": lambda x: False,
    "squad_v2": lambda x: False,
    "drop": lambda x: False,
    "quac": lambda x: False,
    "para_crawl": lambda x: False,
    "wmt16_translate": lambda x: False,
    "wmt14_enfr": lambda x: False,
    "true_case": lambda x: False,
    "fix_punct": lambda x: False,
    "word_segment": lambda x: False,
    "cosmos_qa": cosmos_qa_check,
    "ag_news_subset": ag_news_subset_check,
    "bool_q": lambda x: True,
    "definite_pronoun_resolution": lambda x: True,
    "glue_mrpc": lambda x: True,
    "glue_qqp": lambda x: True,
    "imdb_reviews": imdb_reviews_check,
    "paws_wiki": paws_wiki_check,
    "sentiment140": sentiment140_check,
    "story_cloze": story_cloze_check,
    "copa": copa_check,
    # Technically has multiple choice but ignored because of string parsing issues
    "winogrande": lambda x: False,
    "yelp_polarity_reviews": yelp_polarity_reviews_check,
    "arc": arc_check,
    "anli": anli_check,
    "coqa": lambda x: False,
    "opinion_abstracts_rotten_tomatoes": lambda x: False,
    "opinion_abstracts_idebate": lambda x: False,
    "common_gen": lambda x: False,
    "dart": lambda x: False,
    "e2e_nlg": lambda x: False,
    "web_nlg_en": lambda x: False,
    "wiki_lingua_english_en": lambda x: False,
    "multirc": multirc_check,
    "cb": cb_check,
    "cola": cola_check,
    "sst2": sst2_check,
    "mnli": lambda x: True,
    "qnli": qnli_check,
    "wnli": lambda x: True,
    "snli": snli_check,
    "trec": trec_check,
    "stsb": stsb_check,
    "hellaswag": lambda x: True,
    "piqa": piqa_check,
    "openbookqa": openbookqa_check,
}


def check_if_multiple_choice(data_item):
    inputs = data_item['inputs']
    targets = data_item['targets']
    task_source = data_item['task_source']
    task_name = data_item['task_name']
    template_type = data_item['template_type']
    if '_noopt' in template_type:
        return False
    if 'zs' not in template_type:
        raise ValueError(
            "Template type does not contain zs, do not use this function for non-zs templates")
    for key in list(PATTERNS.keys()):
        if key + ":" in task_name:
            return PATTERNS[key](inputs)
