from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
#                Absolute scoring (rubric-based) evaluation prompts           #
# --------------------------------------------------------------------------- #

ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective "
    "feedback based on specific criteria, ensuring each assessment reflects "
    "the absolute standards set for performance."
)

ABS_USER_PROMPT_TEMPLATE = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{score_rubric}

###Feedback: """

# OpenAI inference parameters for rubric evaluation
SAMPLING_PARAMS_OPENAI = {"max_tokens": 1024, "temperature": 1.0, "top_p": 0.9}


def build_math_prompt(sample, tokenizer: AutoTokenizer = None):
    messages = [{"role": "user", "content": sample["question"]}]
    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt
    else:
        return messages


def build_prefeval_explicit_prompt(sample, tokenizer: AutoTokenizer = None):
    messages = [
        {"role": "system", "content": sample["preference"]},
        {"role": "user", "content": sample["question"]},
    ]
    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt
    else:
        return messages


def build_multifaceted_prompt(sample, tokenizer: AutoTokenizer = None):
    """Construct a chat prompt for a sample from Multifaceted Bench (Janus).

    The dataset follows the schema in Janus' Multifaceted Bench where:
      - `system` contains the preference/system message.
      - `prompt` (or occasionally `question`) contains the user instruction.

    Fallbacks are provided so that the function is robust to slight key name changes.
    """
    # Fallback handling so that the function can also work with samples that use
    # `preference` / `question` like the prefeval dataset.
    system_msg = sample.get("system")
    user_msg = sample.get("prompt")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        return messages
