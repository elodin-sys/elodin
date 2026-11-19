from benchmark import mismatched_sizes, mismatched_witnesses


def run(mode: str, agents: int, digest: str | None, root: str | None = None):
    return {
        "configuration": {"mode": mode, "agents": agents},
        "output_digest": digest,
        "witness": root and {"root": root},
    }


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


def test_repeats_of_one_mode_and_size_must_record_the_same_witness():
    runs = [
        run("local", 2, "a", "x"),
        run("lockstep", 2, "a", "y"),
        run("lockstep", 2, "a", "y"),
        run("lockstep", 4, "b", "y"),
        run("lockstep", 4, "b", "z"),
        run("runtime", 4, None),
    ]
    assert mismatched_witnesses(runs) == [("lockstep", 4)]
