use std::fmt;

/// Extension trait: `iter.join_display(", ")`
pub trait JoinDisplayExt: Iterator + Sized {
    fn join_display(self, sep: &'static str) -> impl fmt::Display
    where
        Self: Clone,
        Self::Item: fmt::Display,
    {
        fmt::from_fn(move |f| {
            let mut it = self.clone();
            if let Some(first) = it.next() {
                write!(f, "{first}")?;
                for item in it {
                    write!(f, "{sep}{item}")?;
                }
            }
            Ok(())
        })
    }
}

impl<I: Iterator> JoinDisplayExt for I {}
