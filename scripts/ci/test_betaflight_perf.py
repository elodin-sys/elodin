#!/usr/bin/env python3
"""Focused tests for the Betaflight metric and bounded confirmation semantics."""

import copy
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import betaflight_perf as perf


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name)
        (self.db / "db_state").write_bytes(b"recorded state")
        cycles = perf.STEPS - 100
        self.summary = {
            "cycles": cycles,
            "steps": cycles,
            "warmup_excluded_ticks": 100,
            "simulation_rate_hz": 8000.0,
            "telemetry_rate_hz": 8000.0,
            "wall_ns": cycles * 200,
        }
        self.summary.update({phase: {"count": cycles, "sum_ns": cycles} for phase in perf.PHASES})
        self.summary["total_cycle"]["sum_ns"] = cycles * 150
        self.summary["real_time_pacing"]["sum_ns"] = cycles * 50
        self.log = (
            "  Total ticks: 119999\n  Sync steps: 123995\n"
            "[C0] lockstep_steps=119995 motor_response=true "
            "max_motor=0.200 takeoff_delta_m=0.500 status=PASS\n"
        )

    def measure(self, summary=None, log=None):
        return perf.measurement(
            self.summary if summary is None else summary,
            self.log if log is None else log,
            self.db,
        )

    def test_work_excludes_pacing_and_accepts_pre_step_warmup_boundary(self):
        self.summary["pre_step"]["count"] -= 1
        result = self.measure()
        self.assertEqual(result["work_ns_per_cycle"], 100)
        slower_sleep = copy.deepcopy(self.summary)
        slower_sleep["real_time_pacing"]["sum_ns"] += slower_sleep["cycles"] * 20
        slower_sleep["total_cycle"]["sum_ns"] += slower_sleep["cycles"] * 20
        slower_sleep["wall_ns"] += slower_sleep["cycles"] * 20
        self.assertEqual(self.measure(slower_sleep)["work_ns_per_cycle"], 100)
        self.assertLess(self.measure(slower_sleep)["achieved_rtf"], result["achieved_rtf"])

    def test_rejects_incomplete_or_missing_metrics(self):
        for key, value in (
            ("cycles", 0),
            ("steps", 119899),
            ("warmup_excluded_ticks", 0),
            ("telemetry_rate_hz", 4000),
            ("wall_ns", 0),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.summary)
                changed[key] = value
                with self.assertRaises(ValueError):
                    self.measure(changed)
        changed = copy.deepcopy(self.summary)
        changed["post_step"]["count"] -= 1
        with self.assertRaises(ValueError):
            self.measure(changed)
        del changed["post_step"]
        with self.assertRaises(KeyError):
            self.measure(changed)
        changed = copy.deepcopy(self.summary)
        changed["world_run"]["sum_ns"] = float("nan")
        with self.assertRaises(ValueError):
            self.measure(changed)

    def test_requires_c0_final_tick_lockstep_no_callback_error_and_db(self):
        for log in (
            self.log.replace("PASS", "FAIL"),
            self.log + self.log,
            self.log.replace("119995", "100"),
            self.log.replace("119999", "119998"),
            self.log + "post_step error: swallowed failure\n",
        ):
            with self.subTest(log=log), self.assertRaises(ValueError):
                self.measure(log=log)
        (self.db / "db_state").unlink()
        with self.assertRaises(ValueError):
            self.measure()

    def test_rejects_swallowed_sensor_read_failure(self):
        with self.assertRaisesRegex(ValueError, "runtime/callback error"):
            self.measure(
                log=self.log + "[SITL] Warning: Could not read sensor data: missing gyro\n"
            )

    def test_comparison_uses_medians_and_work_increase_direction(self):
        def runs(values):
            return [{"work_ns_per_cycle": value, "achieved_rtf": 1.0} for value in values]

        base = runs([100, 100, 10000])
        self.assertFalse(perf.compare(base, runs([110, 110, 999]), 0.10)["regression"])
        result = perf.compare(base, runs([111, 111, 1]), 0.10)
        self.assertTrue(result["regression"])
        self.assertAlmostEqual(result["work_time_increase"], 0.11)
        self.assertFalse(perf.compare(base, runs([90, 90, 9000]), 0.10)["regression"])

    def test_paired_median_handles_bimodal_host_and_preserves_slowdown_sensitivity(self):
        base = [
            {"work_ns_per_cycle": value, "achieved_rtf": 1.0} for value in [112, 147, 151, 114, 114]
        ]
        candidate = [
            {"work_ns_per_cycle": value, "achieved_rtf": 1.0} for value in [143, 150, 144, 116, 113]
        ]
        result = perf.compare(base, candidate, 0.10)
        self.assertGreater(
            result["candidate_work_ns_per_cycle"] / result["base_work_ns_per_cycle"], 1.25
        )
        self.assertAlmostEqual(result["work_time_increase"], 116 / 114 - 1)
        self.assertEqual(len(result["pair_work_time_increases"]), 5)
        self.assertFalse(result["regression"])
        slower = [
            {**run, "work_ns_per_cycle": run["work_ns_per_cycle"] * 1.11} for run in candidate
        ]
        self.assertTrue(perf.compare(base, slower, 0.10)["regression"])
        with self.assertRaises(ValueError):
            perf.compare(base, candidate[:-1], 0.10)


class ControlFlowTests(unittest.TestCase):
    def test_only_regression_gets_one_confirmation(self):
        for outcomes, expected_calls, prefix in (
            ([False], [1], "PASS"),
            ([True, False], [1, 2], "PASS on confirmation"),
            ([True, True], [1, 2], "FAIL"),
        ):
            runner = Mock(side_effect=outcomes)
            self.assertTrue(perf.confirmation(runner).startswith(prefix))
            self.assertEqual([call.args[0] for call in runner.call_args_list], expected_calls)

    def test_hard_failure_is_not_noise_even_during_confirmation(self):
        for outcomes in ([ValueError("bad schema")], [True, ValueError("C0 failed")]):
            runner = Mock(side_effect=outcomes)
            with self.assertRaises(ValueError):
                perf.confirmation(runner)
            self.assertEqual(runner.call_count, len(outcomes))

    def test_process_failure_or_timeout_stops_without_retry_and_reaps_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "eeprom.bin").write_bytes(b"verified eeprom")
            source = root / "source"
            source.mkdir()
            args = SimpleNamespace(
                work=root,
                candidate_source=source,
                eeprom=root / "eeprom.bin",
                output=root,
                addr="127.0.0.1:25000",
                timeout=1,
            )
            inherited = {
                "ELODIN_SIM_WARMUP_TICKS": "119999",
                "ELODIN_PACING_LEAD_US": "1000000",
                "ELODIN_PACING_GRACE_US": "0",
                "ELODIN_PACING_TRACE": "trace.csv",
                "ELODIN_PACING_TRACE_MAX": "100",
                "ELODIN_DETAILED_TIMING": "1",
            }
            for index, outcome in enumerate((7, subprocess.TimeoutExpired("flight", 1))):
                proc = Mock(pid=12345)
                proc.wait.side_effect = [outcome, 0]
                with (
                    patch.object(perf.subprocess, "Popen", return_value=proc) as spawn,
                    patch.object(perf.os, "killpg") as kill,
                    patch.dict(perf.os.environ, inherited),
                ):
                    with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                        perf.confirmation(
                            lambda attempt, index=index: perf.run_flight(
                                args, "candidate", attempt, index
                            )
                        )
                    self.assertEqual(spawn.call_count, 1)
                    child_env = spawn.call_args.kwargs["env"]
                    self.assertEqual(child_env["ELODIN_SIM_WARMUP_TICKS"], "100")
                    self.assertTrue(
                        all(
                            key not in child_env
                            for key in inherited
                            if key != "ELODIN_SIM_WARMUP_TICKS"
                        )
                    )
                    kill.assert_called_once_with(12345, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
