from benchmark import mismatched_sizes


def run(mode: str, agents: int, digest: str | None):
    return {"configuration": {"mode": mode, "agents": agents}, "output_digest": digest}


def test_runs_of_one_size_must_end_in_the_same_state():
    runs = [
        run("runtime", 2, None),
        run("local", 2, "a"),
        run("lockstep", 2, "a"),
        run("lockstep-history", 2, "a"),
        run("local", 4, "b"),
        run("lockstep-history", 4, "c"),
    ]
    assert mismatched_sizes(runs) == [4]
