# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MMLU 5-shot accuracy against an OpenAI-compatible /v1/completions server.

Per subject the 5 dev examples are the few-shot prefix (reduced until it fits
1536 tokens of the gpt-3.5-turbo tokenizer). Plain text completion, temperature
0, one generated token; the prediction is the first character of the stripped
output. Requests are sent concurrently.

usage: mmlu_eval.py --data-dir D --model NAME --port 8000 --concurrency 64
    [--per-subject N] --out result.json
"""

import argparse
import asyncio
import csv
import json
import os
import time

import aiohttp
import tiktoken

CHOICES = ["A", "B", "C", "D"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def format_subject(subject):
    return "".join(" " + e for e in subject.split("_"))


def format_example(row, include_answer=True):
    prompt = row[0]
    k = len(row) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(CHOICES[j], row[j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(row[k + 1])
    return prompt


def gen_prompt(dev_rows, subject, k):
    prompt = (
        "The following are multiple choice questions (with answers) "
        f"about{format_subject(subject)}.\n\n"
    )
    for i in range(k):
        prompt += format_example(dev_rows[i])
    return prompt


def build(data_dir, ntrain, per_subject):
    tok = tiktoken.encoding_for_model("gpt-3.5-turbo")
    subjects = sorted(
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(data_dir, "test"))
        if f.endswith("_test.csv")
    )
    prompts, labels, subj_of = [], [], []
    for subject in subjects:
        dev = read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"))[:ntrain]
        test = read_csv(os.path.join(data_dir, "test", subject + "_test.csv"))
        if per_subject:
            test = test[:per_subject]
        k = ntrain
        shots = gen_prompt(dev, subject, k)
        while len(tok.encode(shots)) > 1536:
            k -= 1
            shots = gen_prompt(dev, subject, k)
        for row in test:
            prompts.append(shots + format_example(row, include_answer=False))
            labels.append(row[-1])
            subj_of.append(subject)
    return prompts, labels, subj_of


async def run(args, prompts):
    url = f"http://{args.host}:{args.port}/v1/completions"
    sem = asyncio.Semaphore(args.concurrency)
    preds = [""] * len(prompts)
    done = 0
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=None)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        async def one(i):
            nonlocal done
            async with sem:
                payload = {
                    "model": args.model,
                    "prompt": prompts[i],
                    "max_tokens": 1,
                    "temperature": 0,
                }
                for attempt in range(3):
                    try:
                        async with session.post(url, json=payload) as r:
                            data = await r.json()
                            text = data["choices"][0]["text"].strip()
                            preds[i] = text[0] if text else ""
                            break
                    except Exception as e:  # retry transient errors
                        if attempt == 2:
                            print("request failed:", i, repr(e)[:200], flush=True)
                done += 1
                if done % 200 == 0:
                    print(
                        f"{done}/{len(prompts)} done, {time.perf_counter() - t0:.0f}s",
                        flush=True,
                    )

        await asyncio.gather(*(one(i) for i in range(len(prompts))))
    return preds, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--ntrain", type=int, default=5)
    ap.add_argument("--per-subject", type=int, default=0, help="0 = all questions")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts, labels, subj_of = build(args.data_dir, args.ntrain, args.per_subject)
    print(f"{len(prompts)} questions, {len(set(subj_of))} subjects", flush=True)
    preds, latency = asyncio.run(run(args, prompts))
    cors = [p == label for p, label in zip(preds, labels)]
    per = {}
    for s, c in zip(subj_of, cors):
        per.setdefault(s, []).append(c)
    for s, v in per.items():
        print(f"subject: {s}, #q:{len(v)}, acc: {sum(v) / len(v):.3f}")
    acc = sum(cors) / len(cors)
    print(f"Total latency: {latency:.3f}")
    print(f"Average accuracy: {acc:.3f}")
    with open(args.out, "w") as f:
        json.dump(
            {
                "model": args.model,
                "num_requests": len(prompts),
                "accuracy": acc,
                "latency": latency,
                "concurrency": args.concurrency,
                "empty_preds": sum(1 for p in preds if p == ""),
                "per_subject": {s: sum(v) / len(v) for s, v in per.items()},
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
