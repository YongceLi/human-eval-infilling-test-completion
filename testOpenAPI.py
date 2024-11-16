import argparse
from openai import OpenAI
from human_eval_infilling.data import write_jsonl, read_problems
from tqdm import tqdm
client = OpenAI()

SYSTEM_PROMPT = "You are a helpful code copilot agent."
INSTRUCTION_PROMPT = (
    "Your task is to complete the code"
    " snippet by filling in the missing code between the provided prefix and"
    " suffix. Ensure correct indentation and line breaks. The user will directly"
    " insert your output into the code, so output only the missing code.\n\n"
)

def construct_few_shot(prefix, suffix, solution):
    return construct_prompt(prefix, suffix) + '<code>' + solution + '</code>\n\n\n'

def construct_prompt(prefix, suffix):
    return f"Prefix:\n<code>{prefix}</code>\n\nSuffix:\n<code>{suffix}</code>\n\nMissing Code:\n"

def remove_overlap(prefix, middle, suffix):
    # Remove overlap with prefix
    for i in range(len(middle)):
        if middle[:i+1] == prefix[-(i+1):]:
            middle = middle[i+1:]
            break
    
    # Remove overlap with suffix
    for i in range(len(middle)):
        if middle[-(i+1):] == suffix[:i+1]:
            middle = middle[:-(i+1)]
            break
    
    return middle

def generate_one_completion(prefix, suffix, system_prompt = "", few_shot_examples = ""):
    prompt = few_shot_examples + construct_prompt(prefix, suffix)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": INSTRUCTION_PROMPT + prompt
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

    problems = read_problems(benchmark_name=benchmark_name)

    few_shot_example = ""
    for i, task_id in enumerate(problems):
        if i < 300 + num_fewshot and i > 300:
            prefix = problems[task_id]["prompt"]
            suffix = problems[task_id]["suffix"]
            solution = problems[task_id]["canonical_solution"]
            few_shot_example += construct_few_shot(prefix, suffix, solution)
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
    args = parser.parse_args()

    main(args) 
