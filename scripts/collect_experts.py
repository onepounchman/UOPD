"""One teacher rollout per task, with exact source prompts and successful replay checks."""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import time

def load_utils(name):
    return importlib.import_module(f'trinity.common.workflows.envs.UOPD.{name}.utils')


def make_prompt(u, name, obs, info, history, task, turn, env):
    formatted = u.format_observation(obs)
    if name == 'alfworld':
        admissible = info.get('admissible_commands', [])
        if admissible and isinstance(admissible[0], list):
            admissible = admissible[0]
        actions = '\n '.join(repr(a) for a in admissible if a != 'help')
        # Match the workflow's explicit single-quote formatting, including apostrophes.
        actions = '\n '.join(f"'{a}'" for a in admissible if a != 'help')
        if not history:
            return u.ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=formatted, admissible_actions=actions)
        return u.ALFWORLD_TEMPLATE.format(task_description=task, step_count=turn,
            history_length=min(u.HISTORY_LENGTH, len(history)), action_history='\n'.join(history[-u.HISTORY_LENGTH:]),
            current_step=turn + 1, current_observation=formatted, admissible_actions=actions)
    actions = u._format_available_actions(env.get_available_actions())
    if len(history) < u.HISTORY_LENGTH:
        return u.WEBSHOP_TEMPLATE_NO_HIS.format(task_description=task, current_observation=formatted, available_actions=actions)
    return u.WEBSHOP_TEMPLATE.format(task_description=task, step_count=turn,
        history_length=min(u.HISTORY_LENGTH, len(history)), action_history='\n'.join(history[-u.HISTORY_LENGTH:]),
        current_step=turn + 1, current_observation=formatted, available_actions=actions)


def replay(u, name, env, task, actions, expected):
    if name == 'alfworld':
        obs, info = env.reset()
    else:
        env.reset(session=task['task_id'])
    done = False
    for action in actions:
        if done:
            raise RuntimeError('Replay ended before the last expert action')
        if name == 'webshop':
            valid, error = u.validate_action(action, env.get_available_actions())
            if not valid:
                raise RuntimeError('Invalid expert replay: ' + str(error))
        obs, reward, done, info = env.step(action)
    score = float(bool(info.get('won', False))) if name == 'alfworld' else float(reward)
    if not done or abs(score - expected) > 1e-6:
        raise RuntimeError(f'Expert replay failed: done={done}, score={score}, expected={expected}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--env', choices=['alfworld', 'webshop'], required=True)
    ap.add_argument('--tasks', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world-size', type=int, default=1)
    ap.add_argument('--teacher')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    name, rank = args.env, args.rank
    if not 0 <= rank < args.world_size:
        ap.error('Expected 0 <= rank < world-size')
    cfg = {
        'model_path': args.teacher or ('langfeng01/GiGPO-Qwen2.5-7B-Instruct-' + ('ALFWorld' if name == 'alfworld' else 'WebShop')),
        'max_model_len': 20480 if name == 'alfworld' else 8192,
        'max_turns': 50 if name == 'alfworld' else 15,
        'key': 'game_file' if name == 'alfworld' else 'task_id',
    }
    meta = {'seed': args.seed, 'temperature': 0.0, 'max_tokens': 512}
    out = args.output / 'shards' / str(rank)
    out.mkdir(parents=True, exist_ok=False)
    from vllm import LLM, SamplingParams
    import torch
    assert torch.cuda.device_count() == 1
    started = time.time()
    u = load_utils(name)
    llm = LLM(model=cfg['model_path'], dtype='bfloat16', tensor_parallel_size=1,
              trust_remote_code=True, max_model_len=cfg['max_model_len'], seed=meta['seed'],
              gpu_memory_utilization=0.8, disable_log_stats=True)
    tok = llm.get_tokenizer()
    random.seed(meta['seed'])
    env = u._create_webshop_env() if name == 'webshop' else None
    identity = {'rank': rank, 'world_size': args.world_size, 'teacher': cfg['model_path'],
                'seed': args.seed, 'temperature': 0.0,
                'started_at': datetime.datetime.now().astimezone().isoformat()}
    if env is not None:
        state = {'goals': env.server.goals, 'prices': env.server.product_prices}
        identity['environment_sha256'] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    (out / 'worker.json').write_text(json.dumps(identity, indent=2) + '\n')
    rows = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()][rank::args.world_size]
    successes = 0
    print(f'READY environment={name} rank={rank} tasks={len(rows)}', flush=True)
    try:
        with (out / 'attempts.jsonl').open('x') as attempts, (out / 'traces.jsonl').open('x') as traces, (out / 'teacher_actions.jsonl').open('x') as experts:
            for index, task in enumerate(rows):
                task_start = time.time()
                key = task[cfg['key']]
                task_key = str(key).split('/json_2.1.1/')[-1]
                # A stable task-derived seed, independent of node, rank and scheduling.
                seed = (meta['seed'] + int(hashlib.sha256((name + ':' + task_key).encode()).hexdigest()[:8], 16)) % (2**31)
                sp = SamplingParams(temperature=meta['temperature'], top_p=1.0, max_tokens=meta['max_tokens'], n=1, seed=seed)
                if name == 'alfworld':
                    env = u._create_alfworld_env(key)
                    obs, info = env.reset()
                    description = u._extract_task(obs)
                else:
                    env.reset(session=key)
                    obs, info = env.observation, {}
                    description = u._extract_task_description(obs)
                initial_observation = obs
                history, actions, steps = [], [], []
                score, done = 0.0, False
                try:
                    for turn in range(cfg['max_turns']):
                        content = make_prompt(u, name, obs, info, history, description, turn, env)
                        prompt = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
                        response = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].text
                        action = u.parse_action(response)
                        history.append(u._format_history(u.format_observation(obs), turn + 1, action))
                        record = {'turn': turn + 1, 'observation': obs, 'prompt': content, 'response': response, 'action': action}
                        if name == 'webshop':
                            valid, error = u.validate_action(action, env.get_available_actions())
                            record['valid_action'] = bool(valid)
                            if valid:
                                actions.append(action)
                                obs, reward, done, info = env.step(action)
                            else:
                                obs, reward, done, info = error, 0.0, False, {}
                        else:
                            actions.append(action)
                            obs, reward, done, info = env.step(action)
                        record.update(reward=float(reward), done=bool(done))
                        steps.append(record)
                        if done:
                            score = float(bool(info.get('won', False))) if name == 'alfworld' else float(reward)
                            break
                    success = score >= 1.0
                    record = {cfg['key']: key, 'attempt': 1, 'seed': seed, 'temperature': meta['temperature'],
                              'score': score, 'success': success, 'done': bool(done), 'turns': len(steps),
                              'actions': actions, 'rank': rank, 'elapsed_seconds': round(time.time() - task_start, 3)}
                    # Preserve the completed rollout even if replay verification detects a problem.
                    traces.write(json.dumps({**record, 'initial_observation': initial_observation, 'steps': steps}) + '\n')
                    traces.flush()
                    if success:
                        if not actions:
                            raise RuntimeError('Successful rollout has no actions')
                        replay(u, name, env, task, actions, score)
                        record['replay_verified'] = True
                        experts.write(json.dumps({**task, 'actions': actions, 'score': score, 'replay_verified': True}) + '\n')
                        experts.flush()
                        successes += 1
                    else:
                        record['replay_verified'] = False
                    attempts.write(json.dumps(record) + '\n')
                    attempts.flush()
                finally:
                    if name == 'alfworld':
                        env.close()
                        env = None
                print(f'PROGRESS {name} rank={rank} tasks={index+1}/{len(rows)} successful={successes} turns={len(steps)} seconds={time.time()-task_start:.1f}', flush=True)
    finally:
        if env is not None:
            env.close()
    (out / 'summary.json').write_text(json.dumps({'complete': True, 'tasks': len(rows), 'attempts': len(rows),
        'successful': successes, 'elapsed_seconds': time.time() - started}, indent=2) + '\n')
    print(f'COMPLETE {name} rank={rank} tasks={len(rows)} successful={successes}', flush=True)


if __name__ == '__main__':
    main()
