use std::{str::FromStr, time::Duration};

use crate::Offset;
use impeller2::types::Timestamp;
use jiff::Span;

peg::parser! {
    grammar offset_parser() for str {

        rule _ = quiet!{[' ' | '\n' | '\t']*}
        rule parse_span() -> Span
            = str:$([_]+) {? str.parse().or(Err("invalid duration")) }

        rule zero() -> Span
            = "0" ['s'|'m'|'h']? { Span::new() }

        rule span() -> Span
            = zero() / parse_span()

        rule epoch() -> hifitime::Epoch
            = str:$([_]+) {? str.parse().or(Err("invalid epoch")) }

        rule sign() -> i64
            = "+" { 1 }
            / "-" { -1 }
        rule start() -> Offset
            = "+" _  span:span()  {? span_to_duration(span).map(Offset::Earliest).or(Err("invalid duration")) }
        rule end() -> Offset
            = "-" _ span:span()  {? span_to_duration(span).map(Offset::Latest).or(Err("invalid duration")) }

        rule fixed() -> Offset
            = "=" _ epoch:epoch()  { Offset::Fixed(Timestamp::from(epoch)) }

        pub rule offset() -> Offset
            = start() / end() / fixed()
    }
}

impl FromStr for Offset {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        offset_parser::offset(s).map_err(|_| ())
    }
}

fn span_to_duration(span: Span) -> Result<Duration, jiff::Error> {
    Ok(Duration::from_nanos(
        span.total(jiff::Unit::Nanosecond)? as u64
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_parse() {
        let o: Offset = "+ 0s".parse().unwrap();
        assert_eq!(o, Offset::Earliest(Duration::from_secs(0)));
        let o: Offset = "+ 20s".parse().unwrap();
        assert_eq!(o, Offset::Earliest(Duration::from_secs(20)));
        let o: Offset = "- 20s".parse().unwrap();
        assert_eq!(o, Offset::Latest(Duration::from_secs(20)))
    }
}
