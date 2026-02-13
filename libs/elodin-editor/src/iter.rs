use std::fmt;

/// A wrapper that joins iterator items with a separator for Display formatting
pub struct JoinDisplay<I> {
    iter: I,
    sep: &'static str,
}

impl<I> fmt::Display for JoinDisplay<I>
where
    I: Iterator + Clone,
    I::Item: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut it = self.iter.clone();
        if let Some(first) = it.next() {
            write!(f, "{first}")?;
            for item in it {
                write!(f, "{}{}", self.sep, item)?;
            }
        }
        Ok(())
    }
}

/// Extension trait: `iter.join_display(", ")`
pub trait JoinDisplayExt: Iterator + Sized {
    fn join_display(self, sep: &'static str) -> JoinDisplay<Self>
    where
        Self: Clone,
        Self::Item: fmt::Display,
    {
        JoinDisplay { iter: self, sep }
    }
}

impl<I: Iterator> JoinDisplayExt for I {}
