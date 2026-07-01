import base64
import json
import mimetypes
import os
import random
from datasets import load_dataset


REMINDER = "\nIn your response, please ensure that you take into account our earlier discussion, and provide an answer that is consistent with my preference."


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def shuffle_options(options):
    """
    Note: In the MCQ datasets, the first choice in the JSON file is the correct answer.
    This function shuffles the options randomly while keeping track of the correct answer.

    :param options: List of options where the first option is the correct answer
    :return: Tuple containing the shuffled options and the index of the correct answer
    """
    correct_answer = options[0]
    shuffled_options = random.sample(options, len(options))
    correct_index = shuffled_options.index(correct_answer)

    return shuffled_options, correct_index


def format_options(options):
    """
    Format a list of options into a lettered string.

    :param options: List of option strings
    :return: Formatted string with lettered options
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    formatted_options = []

    for i, option in enumerate(options):
        formatted_options.append(f"{letters[i]}. {option}")
    return "\n".join(formatted_options)


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def get_prefeval_mcq_question(options):
    """
    Generate the question format with provided options.

    :param options: List of options to be included in the question format
    :return: A formatted MCQ question string
    """
    formatted_options = format_options(options)  # Define formatted_options here

    mcq_question_format = f"""
I'm trying to decide on this and here are 4 options for my query: \n{formatted_options}\nNow, I'd like you to pick one of them as your top recommendation for me.
Important instructions for your response:
1. Choose only one option (A, B, C, or D) that best matches my preferences.
2. Your answer must be one of these options.
3. Don't say things like "I can't choose" or suggest alternatives not listed.
4. Answer example: you should first explain your choice and then give me your answer in this exact format: <choice>X</choice>, where X can be A, B, C, or D.
    """

    return mcq_question_format.strip()


def get_dataset(data_path, split="train"):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, split="train")
    elif data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files=data_path, split="train")
    else:
        data = load_dataset(data_path, split=split)
    return data


def normalize_dataset(example, idx, args, tokenizer):
    dataset_type = args.dataset_type

    if dataset_type.startswith("prefeval"):
        if args.multi_turn_rounds > 0:
            multi_turn_data = json.load(open(args.multi_turn_data_path))
            multi_turn_convs = []
            for data in multi_turn_data:
                multi_turn_convs.extend(data["conversation"])

    if dataset_type == "math":
        MATH_PREFERENCES = {
            "full_math": "Act as a rigorous mathematician. Please think step-by-step to solve this problem. Explicitly state your reasoning for each calculation and verify the logic before providing the final answer. After solving the problem, double-check your result by working backwards or using a different method to ensure the answer is correct.",
            "brief_math": "Solve this math problem step by step.",
            "unrelated": "You are a creative fiction writer. Use vivid imagery, metaphors, and an engaging narrative voice. Focus on character development and emotional depth in your writing.",
        }
        preference_type = getattr(args, "preference_type", "full_math")
        preference = MATH_PREFERENCES.get(preference_type, MATH_PREFERENCES["full_math"])
        question = example["problem"]
        answer = example["answer"]
        non_pref_messages = [{"role": "user", "content": question}]
        pref_messages = [{"role": "system", "content": preference}] + non_pref_messages
        non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)
        pref_prompt = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)
        if args.with_preference:
            prompt = pref_prompt
        else:
            prompt = non_pref_prompt

    elif dataset_type == "prefeval_choice":
        preference = example["preference"]
        question = example["question"]
        answer = ""

        options_data = load_jsonl(args.options_data_path)
        options = options_data[idx]["classification_task_options"]
        shuffled_options, correct_index = shuffle_options(options)
        example["shuffled_options"] = shuffled_options
        example["correct_index"] = correct_index

        convs = example["conversation"]
        preference_messages = [
            dict(role="user", content=convs["query"]),
            dict(role="assistant", content=convs["assistant_options"]),
            dict(role="user", content=convs["user_selection"]),
            dict(role="assistant", content=convs["assistant_acknowledgment"]),
        ]
        messages = []
        if args.multi_turn_rounds > 0:
            messages.extend(multi_turn_convs[:args.multi_turn_rounds])
        
        if args.use_reminder:
            user_query = question + REMINDER
        else:
            user_query = question
        user_query = user_query + "\n" + get_prefeval_mcq_question(options=shuffled_options)
        messages.append(dict(role="user", content=user_query))
        
        if args.preference_type == "system":
            pref_messages = [dict(role="system", content=preference)] + preference_messages + messages
            non_pref_messages = preference_messages + messages
            pref_prompt = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)
            non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)
            prompt = non_pref_prompt
        elif args.preference_type == "implicit":
            pref_messages = preference_messages + messages
            pref_prompt = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)
            non_pref_messages = messages
            non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)
            prompt = pref_prompt
    
    elif dataset_type == "prefeval_explicit":
        preference = example["preference"]
        question = example["question"]
        answer = ""
        
        messages = []
        if args.multi_turn_rounds > 0:
            messages.extend(multi_turn_convs[:args.multi_turn_rounds])
        
        if args.use_reminder:
            user_query = question + REMINDER
        else:
            user_query = question
        messages.append(dict(role="user", content=user_query))
        
        non_pref_messages = messages
        non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)

        pref_messages = [dict(role="system", content=preference)] + messages
        pref_prompt = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)
        prompt = pref_prompt

    elif dataset_type == "prefeval_implicit":
        preference = example["preference"]
        question = example["question"]
        answer = ""

        convs = example["conversation"]
        preference_messages = []
        # sort by key and flatten
        for i in range(len(convs)):
            turn = convs[str(i)]
            if turn:
                if "user" in turn:
                    preference_messages.append(dict(role="user", content=turn["user"]))
                if "assistant" in turn:
                    preference_messages.append(dict(role="assistant", content=turn["assistant"]))
        
        messages = []
        if args.multi_turn_rounds > 0:
            messages.extend(multi_turn_convs[:args.multi_turn_rounds])
        
        if args.use_reminder:
            user_query = question + REMINDER
        else:
            user_query = question
        messages.append(dict(role="user", content=user_query))
        
        non_pref_messages = preference_messages + messages
        non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)

        pref_messages = [dict(role="system", content=preference)] + preference_messages + messages
        pref_prompt = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)
        prompt = non_pref_prompt

    elif dataset_type == "custom_text":
        preference = example.get("preference") or getattr(args, "default_system_prompt", "")
        question = example["question"]
        answer = example.get("answer", "")

        non_pref_messages = [{"role": "user", "content": question}]
        if preference:
            pref_messages = [{"role": "system", "content": preference}] + non_pref_messages
            pref_prompt = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)
        else:
            pref_messages = non_pref_messages
            pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)

        non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)
        use_preference = bool(getattr(args, "with_preference", False)) and bool(preference)
        prompt = pref_prompt if use_preference else non_pref_prompt

    elif dataset_type == "ping_pong":
        # Simplified for ping_pong, assuming single prompt structure initially
        character = example["character"]
        situation = example["situation"]
        
        # Load templates. Caching them would be better for performance.
        with open(args.templates.interrogator_user, "r") as f:
            interrogator_user_template = f.read()
        with open(args.templates.player_character, "r") as f:
            player_system_template = f.read()
        
        # Prepare initial messages and prompts.
        # The multi-turn logic will handle history.
        # For turn 0, history is empty.
        messages = [] 
        if character.get("initial_message"):
            messages.append({"role": "assistant", "content": character["initial_message"]})

        # For the player
        from jinja2 import Template
        system_prompt = Template(player_system_template).render(character=character)
        player_messages = [{"role": "system", "content": system_prompt}] + messages
        prompt = tokenizer.apply_chat_template(player_messages, add_generation_prompt=True, tokenize=False)

        # For the interrogator. The actual prompt will be constructed turn-by-turn in the runner.
        # We store the template-renderable string in _interrogator_prompt
        # Note: this is a bit of a hack. A cleaner way would be to pass templates around.
        interrogator_prompt_content = Template(interrogator_user_template).render(
            char_summary=character.get("summary", character.get("char_name")),
            situation=situation.get("text", ""),
            messages=[], # placeholder for history
        )
        # We only need the part that will be the "user" content for the interrogator
        example["_interrogator_prompt_template"] = interrogator_user_template
        
        preference = system_prompt
        question = situation.get("text", "")
        answer = ""

        # With preference (system prompt)
        pref_messages = [{"role": "system", "content": system_prompt}] + messages
        pref_prompt = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)

        # Without preference (system prompt)
        non_pref_messages = messages
        if non_pref_messages:
            non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)
        else:
            # The conversation history is empty, so there's no prompt yet.
            # It will be constructed in the runner after the first interrogator turn.
            non_pref_prompt = ""

    elif dataset_type == "mmhal_bench":
        preference = (
            "You are a careful visual assistant. Ground every statement strictly in the image and the question. "
            "If a detail is not clearly visible, say you are unsure or that it is not shown. "
            "Do not guess objects, actions, text, or attributes that are not present. "
            "Prefer admitting uncertainty over making up details. Be concise and factual."
        )
        question = example["question"]
        answer = example["gt_answer"]
        image_path = example.get("image_path")
        if not image_path:
            image_dir = os.path.join(os.path.dirname(args.dataset_path), "images")
            image_file = os.path.basename(example["image_src"])
            image_path = os.path.join(image_dir, image_file)
            example["image_path"] = image_path
        image_url = image_to_data_url(image_path)

        non_pref_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        pref_messages = [
            {"role": "system", "content": [{"type": "text", "text": preference}]}
        ] + non_pref_messages
        pref_prompt = pref_messages
        non_pref_prompt = non_pref_messages
        if args.with_preference:
            prompt = pref_prompt
        else:
            prompt = non_pref_prompt

    elif dataset_type == "multifaceted":
        preference = example["system"]
        question = example["prompt"]
        answer = example["reference_answer"]
        messages = [dict(role="system", content=preference), dict(role="user", content=question)]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        non_pref_messages = [dict(role="user", content=question)]
        pref_prompt = prompt
        non_pref_prompt = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)
    
    example.update({
        "_preference": preference,
        "_question": question,
        "_prompt": prompt,
        "_answer": answer,
        "_pref_prompt": pref_prompt,
        "_non_pref_prompt": non_pref_prompt,
    })
    return example
