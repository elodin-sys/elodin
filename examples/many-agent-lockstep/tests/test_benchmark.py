from benchmark import mismatched_sizes


def run(mode: str, agents: int, digest: str | None, evidence: str | None = None):
    return {
        "configuration": {"mode": mode, "agents": agents},
        "output_digest": digest,
        "evidence_digest": evidence,
    }


def test_runs_of_one_size_must_agree():
    runs = [
        run("runtime", 2, None),
        run("local", 2, "a"),
        run("lockstep", 2, "a"),
        run("lockstep-history", 2, "a", "e"),
        run("local", 4, "b"),
        run("lockstep", 4, "c"),
        run("lockstep-history", 8, "d", "e"),
        run("lockstep-history", 8, "d", "f"),
    ]
    assert mismatched_sizes(runs) == [4, 8]
