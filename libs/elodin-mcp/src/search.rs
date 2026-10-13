//! The `search` tool: one call expression, parsed (never evaluated) and answered from the index.

use std::sync::LazyLock;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::index::Index;

pub const MAX_OUTPUT: usize = 16 * 1024;
pub const DEFAULT_LIMIT: usize = 20;

pub const GRAMMAR: &str = "Pass one expression: runs(), tables(query=None, run=None, limit=20), \
signals(query=None, run=None, limit=20), describe(name), sdk(query=None), or plain text \
(same as signals(\"text\")). Arguments are strings, integers, or None.";

#[derive(Clone, Debug, Deserialize)]
pub struct SdkEntry {
    pub name: String,
    pub signature: String,
    pub summary: String,
    pub example: String,
}

pub static SDK: LazyLock<Vec<SdkEntry>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("sdk_reference.json")).expect("sdk_reference.json")
});

const QUICKSTART: &[&str] = &[
    "`el` (elodin.analysis) and `ctx` are pre-bound in execute; numpy, polars, pyarrow, matplotlib are installed.",
    "df = el.signal('<signal>', run='<run>').window(start_us=..., end_us=...).to_polars()  # time_us, value",
    "el.table('<table>', run='<run>').select('a', 'b').to_polars()  # joined members; nulls where absent",
    "time_us is int64 microseconds on the recorder clock (not wall time); names are case-insensitive.",
    "el.artifacts.save_table(df, 'name') / save_figure(fig, 'name') save outputs; print() goes to a log.",
    "Assign a small JSON-serializable dict to `result`; it is the only output returned.",
];

#[derive(Clone, Debug, PartialEq)]
enum Arg {
    Str(String),
    Int(i64),
    None,
}

#[derive(Debug, PartialEq)]
struct Call {
    name: String,
    positional: Vec<Arg>,
    keyword: Vec<(String, Arg)>,
}

impl Call {
    fn arg(&self, pos: usize, key: &str) -> Option<&Arg> {
        self.keyword
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .or_else(|| self.positional.get(pos))
    }
    fn str(&self, pos: usize, key: &str) -> Result<Option<String>, String> {
        match self.arg(pos, key) {
            None | Some(Arg::None) => Ok(None),
            Some(Arg::Str(s)) => Ok(Some(s.clone())),
            Some(Arg::Int(_)) => Err(format!("{}(): `{key}` must be a string", self.name)),
        }
    }
    fn limit(&self, pos: usize) -> Result<usize, String> {
        match self.arg(pos, "limit") {
            None | Some(Arg::None) => Ok(DEFAULT_LIMIT),
            Some(Arg::Int(n)) if *n > 0 => Ok(*n as usize),
            _ => Err(format!(
                "{}(): `limit` must be a positive integer",
                self.name
            )),
        }
    }
}

/// Parse `name(arg, key=arg, ...)`. `Ok(None)` means the text isn't a call.
fn parse(text: &str) -> Result<Option<Call>, String> {
    let text = text.trim();
    let Some(open) = text.find('(') else {
        return Ok(None);
    };
    let name = text[..open].trim();
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Ok(None);
    }
    let rest = &text[open + 1..];
    let close = rest
        .rfind(')')
        .ok_or_else(|| format!("missing ')' in {text:?}"))?;
    if !rest[close + 1..].trim().is_empty() {
        return Err("expected exactly one call".into());
    }
    let mut chars = rest[..close].chars().peekable();
    let mut call = Call {
        name: name.to_string(),
        positional: vec![],
        keyword: vec![],
    };
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        let Some(&c) = chars.peek() else { break };
        let mut key = None;
        if c.is_ascii_alphabetic() || c == '_' {
            let mut word = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    word.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
            if chars.peek() == Some(&'=') {
                chars.next();
                key = Some(word);
            } else if word == "None" {
                call.positional.push(Arg::None);
                skip_comma(&mut chars)?;
                continue;
            } else {
                return Err(format!("unexpected identifier `{word}`; quote strings"));
            }
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
        }
        let value = parse_value(&mut chars)?;
        match key {
            Some(k) => call.keyword.push((k, value)),
            None if !call.keyword.is_empty() => {
                return Err("positional argument after keyword argument".into());
            }
            None => call.positional.push(value),
        }
        skip_comma(&mut chars)?;
    }
    Ok(Some(call))
}

fn skip_comma(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<(), String> {
    while chars.peek().is_some_and(|c| c.is_whitespace()) {
        chars.next();
    }
    match chars.next() {
        None | Some(',') => Ok(()),
        Some(c) => Err(format!("expected ',' but found {c:?}")),
    }
}

fn parse_value(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<Arg, String> {
    match chars.peek() {
        Some(&q) if q == '"' || q == '\'' => {
            chars.next();
            let mut s = String::new();
            loop {
                match chars.next() {
                    None => return Err("unterminated string".into()),
                    Some('\\') => s.extend(chars.next()),
                    Some(c) if c == q => return Ok(Arg::Str(s)),
                    Some(c) => s.push(c),
                }
            }
        }
        Some(c) if c.is_ascii_digit() || *c == '-' => {
            let mut s = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == '-' || c == '_' {
                    s.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            s.replace('_', "")
                .parse()
                .map(Arg::Int)
                .map_err(|_| format!("bad integer {s:?}"))
        }
        Some(&c) if c.is_ascii_alphabetic() => {
            let mut s = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() {
                    s.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            if s == "None" {
                Ok(Arg::None)
            } else {
                Err(format!("unexpected identifier `{s}`; quote strings"))
            }
        }
        other => Err(format!("expected a value, found {other:?}")),
    }
}

/// Rank `name` against `query`: lower is better, `None` is no match.
pub fn rank(query: &str, name: &str) -> Option<u8> {
    let q = query.trim().to_lowercase();
    let n = name.to_lowercase();
    if q.is_empty() {
        return Some(6);
    }
    if n == q {
        return Some(0);
    }
    let last = n.rsplit('.').next().unwrap_or(&n);
    if last == q {
        return Some(1);
    }
    if n.starts_with(&q) || last.starts_with(&q) {
        return Some(2);
    }
    if n.contains(&q) {
        return Some(3);
    }
    let tokens: Vec<&str> = q
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    if !tokens.is_empty() && tokens.iter().all(|t| n.contains(t)) {
        return Some(4);
    }
    let fuzzy = strsim::jaro_winkler(&q, &n).max(strsim::jaro_winkler(&q, last));
    (fuzzy >= 0.85).then_some(5)
}

fn similarity(query: &str, name: &str) -> i64 {
    -((strsim::jaro_winkler(&query.to_lowercase(), name) * 1000.0) as i64)
}

/// Names ranked against `query`; ties go to the closer name, then to populated ones.
pub fn ranked(query: &str, names: &std::collections::BTreeMap<String, bool>) -> Vec<String> {
    let mut hits: Vec<(u8, i64, bool, &String)> = names
        .iter()
        .filter_map(|(n, populated)| {
            rank(query, n).map(|r| (r, similarity(query, n), !populated, n))
        })
        .collect();
    hits.sort();
    hits.into_iter().map(|(.., n)| n.clone()).collect()
}

/// Up to five close names, for did-you-mean.
pub fn suggestions(query: &str, names: &std::collections::BTreeMap<String, bool>) -> Vec<String> {
    let mut scored: Vec<(u8, i64, &String)> = names
        .keys()
        .map(|n| (rank(query, n).unwrap_or(u8::MAX), similarity(query, n), n))
        .collect();
    scored.sort();
    scored
        .into_iter()
        .take(5)
        .map(|(.., n)| n.clone())
        .collect()
}

/// Answer one search expression. `None` returns the overview.
pub fn search(index: &Index, code: Option<&str>, limit: Option<usize>) -> Value {
    let code = code.map(str::trim).filter(|c| !c.is_empty());
    let Some(code) = code else {
        return overview(index);
    };
    let call = match parse(code) {
        Ok(Some(call)) => call,
        Ok(None) => Call {
            name: "signals".into(),
            positional: vec![Arg::Str(code.into())],
            keyword: vec![],
        },
        Err(e) => return json!({"error": e, "grammar": GRAMMAR}),
    };
    match answer(index, &call, limit) {
        Ok(v) => v,
        Err(e) => json!({"error": e, "grammar": GRAMMAR}),
    }
}

fn answer(index: &Index, call: &Call, limit: Option<usize>) -> Result<Value, String> {
    let results: Vec<Value> = match call.name.as_str() {
        "runs" => index.runs.iter().map(|r| index.run_card(r)).collect(),
        "signals" | "tables" => {
            let query = call.str(0, "query")?.unwrap_or_default();
            let run = call.str(1, "run")?;
            if let Some(r) = &run
                && !index.runs.iter().any(|x| x.name.eq_ignore_ascii_case(r))
            {
                let names = index
                    .runs
                    .iter()
                    .map(|r| r.name.clone())
                    .collect::<Vec<_>>();
                return Err(format!("unknown run {r:?}; runs: {names:?}"));
            }
            let n = limit.unwrap_or(call.limit(2)?);
            let signals = call.name == "signals";
            let names = if signals {
                index.signal_names(run.as_deref())
            } else {
                index.table_names(run.as_deref())
            };
            let hits = ranked(&query, &names);
            if hits.is_empty() {
                return Ok(json!({
                    "results": [],
                    "truncated": false,
                    "hint": format!("no {} match {query:?}", call.name),
                    "did_you_mean": suggestions(&query, &names),
                }));
            }
            let total = hits.len();
            let cards: Vec<Value> = hits
                .iter()
                .take(n)
                .filter_map(|h| {
                    if signals {
                        index.signal_card(h, run.as_deref(), false)
                    } else {
                        index.table_card(h, run.as_deref(), false)
                    }
                })
                .collect();
            if total > n {
                return Ok(capped(
                    cards,
                    Some(format!(
                        "{total} matches; showing {n}. Narrow the query or raise limit."
                    )),
                ));
            }
            cards
        }
        "describe" => {
            let name = call.str(0, "name")?.ok_or("describe(name) needs a name")?;
            if let Some(c) = index.signal_card(&name, None, true) {
                vec![c]
            } else if let Some(c) = index.table_card(&name, None, true) {
                vec![c]
            } else if let Some(r) = index
                .runs
                .iter()
                .find(|r| r.name.eq_ignore_ascii_case(&name))
            {
                vec![index.run_card(r)]
            } else {
                let mut names = index.signal_names(None);
                names.extend(index.table_names(None));
                return Ok(json!({
                    "results": [],
                    "truncated": false,
                    "hint": format!("nothing named {name:?}"),
                    "did_you_mean": suggestions(&name, &names),
                }));
            }
        }
        "sdk" => {
            let query = call.str(0, "query")?.unwrap_or_default().to_lowercase();
            SDK.iter()
                .filter(|e| {
                    query.is_empty()
                        || e.name.to_lowercase().contains(&query)
                        || e.summary.to_lowercase().contains(&query)
                })
                .map(|e| {
                    json!({"kind": "sdk", "name": e.name, "signature": e.signature,
                           "summary": e.summary, "example": e.example})
                })
                .collect()
        }
        other => return Err(format!("unknown function `{other}`")),
    };
    Ok(capped(results, None))
}

fn overview(index: &Index) -> Value {
    let runs: Vec<Value> = index.runs.iter().map(|r| index.run_card(r)).collect();
    let mut tables: Vec<Value> = Vec::new();
    for (name, populated) in index.table_names(None) {
        if !populated {
            continue;
        }
        let card = index.table_card(&name, None, true).expect("table exists");
        let columns = card["columns"].as_array().map_or(0, |c| c.len());
        tables.push(json!({"name": name, "columns": columns, "runs": card["runs"]}));
    }
    let results = vec![
        json!({"kind": "overview", "runs": runs}),
        json!({"kind": "tables", "tables": tables}),
        json!({"kind": "quickstart", "lines": QUICKSTART, "grammar": GRAMMAR}),
    ];
    capped(results, None)
}

/// Wrap results, dropping trailing ones past [`MAX_OUTPUT`].
fn capped(results: Vec<Value>, hint: Option<String>) -> Value {
    let mut kept = Vec::new();
    let mut size = 64 + hint.as_ref().map_or(0, |h| h.len());
    let mut truncated = hint.is_some();
    for r in results {
        let len = serde_json::to_string(&r).map_or(0, |s| s.len()) + 1;
        if size + len > MAX_OUTPUT {
            truncated = true;
            break;
        }
        size += len;
        kept.push(r);
    }
    let hint = match (hint, truncated) {
        (Some(h), _) => Some(h),
        (None, true) => Some(format!(
            "output capped at {} KiB; narrow the query or lower limit",
            MAX_OUTPUT / 1024
        )),
        (None, false) => None,
    };
    let mut out = json!({"results": kept, "truncated": truncated});
    if let Some(h) = hint {
        out["hint"] = json!(h);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_calls() {
        let c = parse("signals('accel', run=\"ft28\", limit=5)")
            .unwrap()
            .unwrap();
        assert_eq!(c.name, "signals");
        assert_eq!(c.positional, vec![Arg::Str("accel".into())]);
        assert_eq!(c.keyword[0], ("run".into(), Arg::Str("ft28".into())));
        assert_eq!(c.keyword[1], ("limit".into(), Arg::Int(5)));
        assert_eq!(parse("runs()").unwrap().unwrap().positional, vec![]);
        assert_eq!(
            parse("tables(None, None)").unwrap().unwrap().positional,
            vec![Arg::None, Arg::None]
        );
        assert!(parse("accel pressure").unwrap().is_none());
        assert!(parse("signals(accel)").is_err());
        assert!(parse("signals('a')) + 1").is_err());
        assert!(parse("signals(run='x', 'a')").is_err());
    }

    #[test]
    fn ranks() {
        assert_eq!(rank("imu.accel", "IMU.ACCEL"), Some(0));
        assert_eq!(rank("accel", "imu.accel"), Some(1));
        assert_eq!(rank("acc", "imu.accel"), Some(2));
        assert_eq!(rank("mu.acc", "imu.accel"), Some(3));
        assert_eq!(rank("accel imu", "imu.accel"), Some(4));
        assert_eq!(rank("imu.acel", "imu.accel"), Some(5));
        assert_eq!(rank("pressure", "imu.accel"), None);
    }

    #[test]
    fn sdk_reference_loads() {
        assert!(SDK.len() >= 10);
    }
}
