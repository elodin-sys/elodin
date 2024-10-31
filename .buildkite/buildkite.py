import json


def step(label, command, key=None, depends_on=None, env={}, plugins=[], skip=False, agents={}):
    return {
        "label": label,
        "command": command,
        "key": key,
        "depends_on": depends_on,
        "env": env,
        "plugins": plugins,
        "skip": skip,
        "agents": agents,
    }


def group(name, is_gha=False, key=None, depends_on=None, steps=[]):
    return {
        "group": name,
        "key": key,
        "depends_on": depends_on,
        "steps": steps,
    }


def pipeline(steps=[], env={}):
    return print(json.dumps({"steps": steps, "env": env}))
