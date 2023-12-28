import json

def step(label, command, key = None, depends_on = None, env = {}, plugins = [], condition = None, skip = False):
  return {
    "label": label,
    "command": command,
    "key": key,
    "depends_on": depends_on,
    "env": env,
    "plugins": plugins,
    "if": condition,
    "skip": skip,
  }

def group(name, is_gha = False, key = None, depends_on = None, steps = []):
  return {
    "group": name,
    "key": key,
    "depends_on": depends_on,
    "steps": steps,
    "if": f"build.env(\"TRIGGERED_FROM_GHA\") {'==' if is_gha else '!='} \"1\"",
  }

def pipeline(steps = [], env = {}):
  return print(json.dumps({"steps": steps, "env": env}))
