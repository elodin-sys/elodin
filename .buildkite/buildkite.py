import json


def step(label, command, key=None, depends_on=None, env={}, plugins=[], skip=False, agents={}, if_condition=None):
    result = {
        "label": label,
        "command": command,
        "key": key,
        "depends_on": depends_on,
        "env": env,
        "plugins": plugins,
        "skip": skip,
        "agents": agents,
    }
    if if_condition:
        result["if"] = if_condition
    return result


def group(name, is_gha=False, key=None, depends_on=None, steps=[]):
    return {
        "group": name,
        "key": key,
        "depends_on": depends_on,
        "steps": steps,
    }


def pipeline(steps=[], env={}, agents={}):
    result = {"steps": steps, "env": env}
    if agents:
        result["agents"] = agents
    return print(json.dumps(result))
