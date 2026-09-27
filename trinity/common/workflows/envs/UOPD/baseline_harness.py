"""Current-turn prompts for baseline continuation probes, matching local OPD."""


def alfworld_messages(observation, info, history, task_description, step):
    from trinity.common.workflows.envs.UOPD.alfworld.utils import (
        ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS, HISTORY_LENGTH,
        format_observation,
    )
    admissible = info.get('admissible_commands', [])
    if admissible and isinstance(admissible[0], list):
        admissible = admissible[0]
    actions = '\n '.join(f"'{s}'" for s in admissible if s != 'help')
    fields = dict(current_observation=format_observation(observation),
                  admissible_actions=actions)
    if not history:
        content = ALFWORLD_TEMPLATE_NO_HIS.format(**fields)
    else:
        content = ALFWORLD_TEMPLATE.format(
            **fields, task_description=task_description, step_count=step,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history='\n'.join(history[-HISTORY_LENGTH:]), current_step=step + 1)
    return [{'role': 'user', 'content': content}]


def webshop_messages(observation, available_actions, history, task_description, step):
    from trinity.common.workflows.envs.UOPD.webshop.utils import (
        WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS, HISTORY_LENGTH,
        _format_available_actions, format_observation,
    )
    fields = dict(task_description=task_description,
                  current_observation=format_observation(observation),
                  available_actions=_format_available_actions(available_actions))
    if len(history) < HISTORY_LENGTH:
        content = WEBSHOP_TEMPLATE_NO_HIS.format(**fields)
    else:
        content = WEBSHOP_TEMPLATE.format(
            **fields, step_count=step, history_length=min(HISTORY_LENGTH, len(history)),
            action_history='\n'.join(history[-HISTORY_LENGTH:]), current_step=step + 1)
    return [{'role': 'user', 'content': content}]
