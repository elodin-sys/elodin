use std::{path::PathBuf, time::Duration};

use elodin_types::BitVec;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DurationSeconds};

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
pub struct Report {
    #[serde_as(as = "DurationSeconds<f64>")]
    duration: Duration,
    exitcode: u8,
    root: PathBuf,
    pub summary: Summary,
    tests: Vec<Test>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to parse json: {0}")]
    Json(#[from] serde_json::Error),
}

impl Report {
    pub fn from_json(json: &str) -> Result<Self, Error> {
        serde_json::from_str(json).map_err(Error::from)
    }

    pub fn failed(&self, batch_size: usize, min_sample_number: usize) -> BitVec {
        let mut bitvec = BitVec::from_elem(batch_size, false);
        for test in &self.tests {
            if test.outcome != Outcome::Passed {
                if let Some(sample_number) = test.sample_number() {
                    bitvec.set(sample_number - min_sample_number, true);
                }
            }
        }
        bitvec
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Summary {
    #[serde(default)]
    passed: usize,
    #[serde(default)]
    failed: usize,
    #[serde(default)]
    total: usize,
    #[serde(default)]
    collected: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Test {
    nodeid: String,
    lineno: usize,
    outcome: Outcome,
    keywords: Vec<String>,
}

impl Test {
    fn sample_number(&self) -> Option<usize> {
        let keyword = self.keywords.first()?;
        // Given a keyword like "test_function[0]", extract the sample number in the brackets
        keyword.split('[').last()?.split(']').next()?.parse().ok()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Passed,
    Failed,
    Skipped,
    Error,
    XFailed,
    XPassed,
    FailedExpectation,
    Aborted,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_REPORT: &str = include_str!("../test/report.json");

    #[test]
    fn parse_good_report() {
        let report = Report::from_json(TEST_REPORT).unwrap();
        let failed = report.failed(10, 0);
        assert_eq!(report.exitcode, 1);
        assert!(!failed[0]);
        assert!(failed[4]);
    }
}
