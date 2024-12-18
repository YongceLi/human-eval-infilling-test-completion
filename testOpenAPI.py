import argparse
from openai import OpenAI
from human_eval_infilling.data import write_jsonl, read_problems
from tqdm import tqdm
client = OpenAI()

SYSTEM_PROMPT = "You are a helpful code copilot agent."
INSTRUCTION_PROMPT_MISSING_CODE = (
    "Your task is to complete the code"
    " snippet by filling in the missing code between the provided prefix and"
    " suffix. Ensure correct indentation and line breaks. The user will directly"
    " insert your output into the code, so output only the missing code.\n\n"
)

INSTRUCTION_PROMPT_COMPLETE_CODE = (
    "You are given a code with a part missing; the prefix and the suffix of the"
    " code are provided. Your task is to complete the code snippet by filling in"
    " the missing code between the provided prefix and suffix. Ensure correct"
    " indentation and line breaks. Ensure you do not modify the prefix or the "
    "suffix. Ensure your output code is enclosed between <code> and </code>. "
    "Output the completed code."
)
INSTRUCTION_PROMPT = {"output_missing": INSTRUCTION_PROMPT_MISSING_CODE, 
                      "output_complete": INSTRUCTION_PROMPT_COMPLETE_CODE}

def construct_few_shot(prefix, suffix, solution, mode = "output_missing"):
    assert mode in ["output_missing", "output_complete"]
    if mode == "output_missing":
        return construct_prompt(prefix, suffix) + '<code>' + solution + '</code>\n\n\n'
    if mode == "output_complete":
        return construct_prompt(prefix, suffix, mode = mode) + '<code>' + prefix + solution + suffix + '</code>\n\n\n'

def construct_prompt(prefix, suffix, mode = "output_missing"):
    assert mode in ["output_missing", "output_complete"]
    if mode == "output_missing":
        return f"Prefix:\n<code>{prefix}</code>\n\nSuffix:\n<code>{suffix}</code>\n\nMissing Code:\n"
    if mode == "output_complete":
        return f"Prefix:\n<code>{prefix}</code>\n\nSuffix:\n<code>{suffix}</code>\n\nComplete Code:\n"

def remove_overlap(prefix, output, suffix):
    # Remove overlap with prefix
    for i in range(len(output)):
        if output[:i+1] == prefix[-(i+1):]:
            output = output[i+1:]
            break
    
    # Remove overlap with suffix
    for i in range(len(output)):
        if output[-(i+1):] == suffix[:i+1]:
            output = output[:-(i+1)]
            break
    
    return output

def generate_one_completion(prefix, suffix, system_prompt = "", few_shot_examples = "", mode = "output_missing"):
    prompt = few_shot_examples + construct_prompt(prefix, suffix, mode = mode)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": INSTRUCTION_PROMPT[mode] + prompt
            }
        ]
    )
    generated_answer = response.choices[0].message.content.replace("```python\n", "").replace("```", "")\
        .replace("<code>", "").replace("</code>", "")
    generated_answer = remove_overlap(prefix, generated_answer, suffix)
    return generated_answer

def main(args):
    benchmark_name = args.benchmark
    num_samples_per_task = args.n_samples
    num_fewshot = args.n_fewshot
    mode = args.mode

    problems = read_problems(benchmark_name=benchmark_name)

    few_shot_example = ""
    for i, task_id in enumerate(problems):
        if i < 300 + num_fewshot and i > 300:
            prefix = problems[task_id]["prompt"]
            suffix = problems[task_id]["suffix"]
            solution = problems[task_id]["canonical_solution"]
            few_shot_example += construct_few_shot(prefix, suffix, solution, mode = mode)
        else:
            continue
    
    if args.debug:
        test_number, test_problem = next(iter(problems.items()))
        completion = generate_one_completion(test_problem["prompt"],
                                             test_problem["suffix"],
                                             system_prompt=SYSTEM_PROMPT,
                                             few_shot_examples=few_shot_example)
        
        print(f"prefix: {test_problem['prompt']}\n\n suffix: {test_problem['suffix']}\n\n output: {completion}\n\n" )

    else:
        samples = [
            dict(task_id = task_id, 
                 completion = generate_one_completion(problems[task_id]["prompt"],
                                                      problems[task_id]["suffix"],
                                                      system_prompt=SYSTEM_PROMPT,
                                                      few_shot_examples=few_shot_example)
                )
            for task_id in tqdm(problems)
            for _ in range(num_samples_per_task)
        ]

        write_jsonl(f"samples_{benchmark_name.replace('-', '_')}.jsonl", samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--benchmark', type=str, choices=['single-line', 'multi-line', 'random-span', 'random-span-light'], help='choose the benchmark to test', required=True)
    parser.add_argument('--n_samples', type=int, default=1, help='choose number of samples to test per question')
    parser.add_argument('--n_fewshot', type=int, default=0, help='choose number of few shot examples to be appended')
    parser.add_argument('--debug', action="store_true", help='debug mode, only test on one sample')
    parser.add_argument('--mode', type=str, default='output_missing', choices=['output_missing', 'output_complete'], help='choose the output format: whether output only missing code/complete code')
    args = parser.parse_args()

    main(args) 
