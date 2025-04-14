#![cfg_attr(not(test), allow(dead_code, unused_macros))]

use mycelium_util::fmt;

macro_rules! span {
    ($level:expr, $($arg:tt)+) => {
        crate::trace::Span {
            span_01: {
                use tracing::Level;
                tracing::span!($level, $($arg)+)
            },
        }
    }
}

macro_rules! trace {
    ($($arg:tt)+) => {
            tracing::trace!($($arg)+)

    };
}

macro_rules! debug {
    ($($arg:tt)+) => {
        tracing::debug!($($arg)+)
    };
}

#[cfg(test)]
macro_rules! info {
    ($($arg:tt)+) => {
        tracing::info!($($arg)+)
    };
}

macro_rules! trace_span {
    ($($arg:tt)+) => {
        span!(Level::TRACE, $($arg)+)
    };
}

#[allow(unused_macros)]
macro_rules! debug_span {
    ($($arg:tt)+) => {
        span!(Level::DEBUG, $($arg)+)
    };
}

#[cfg(all(not(test), not(maitake_ultraverbose)))]
macro_rules! test_dbg {
    ($e:expr) => {
        $e
    };
}

#[cfg(any(test, maitake_ultraverbose))]
macro_rules! test_dbg {
    ($e:expr) => {
        match $e {
            e => {
                debug!(
                    location = %core::panic::Location::caller(),
                    "{} = {:?}",
                    stringify!($e),
                    &e
                );
                e
            }
        }
    };
}

#[cfg(all(not(test), not(maitake_ultraverbose)))]
macro_rules! test_debug {
    ($($args:tt)+) => {};
}

#[cfg(any(test, maitake_ultraverbose))]
macro_rules! test_debug {
    ($($args:tt)+) => {
        debug!($($args)+);
    };
}

#[cfg(all(not(test), not(maitake_ultraverbose)))]
macro_rules! test_trace {
    ($($args:tt)+) => {};
}

#[cfg(any(test, maitake_ultraverbose))]
macro_rules! test_trace {
    ($($args:tt)+) => {
        trace!($($args)+);
    };
}

#[derive(Clone)]
pub(crate) struct Span {
    pub(crate) span_01: tracing::Span,
}

impl Span {
    #[inline(always)]
    pub(crate) const fn none() -> Self {
        Span {
            span_01: tracing::Span::none(),
        }
    }

    #[inline]
    pub(crate) fn enter(&self) -> Entered<'_> {
        Entered {
            _enter_01: self.span_01.enter(),
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub(crate) fn entered(self) -> EnteredSpan {
        EnteredSpan {
            _enter_01: self.span_01.entered(),
        }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const TRACING_FIELD: &str = "tracing";

        let mut s = f.debug_struct("Span");

        if let Some(id) = self.span_01.id() {
            s.field(TRACING_FIELD, &id.into_u64());
        } else {
            s.field(TRACING_FIELD, &fmt::display("<none>"));
        }

        s.finish()
    }
}

#[derive(Debug)]
pub(crate) struct Entered<'span> {
    _enter_01: tracing::span::Entered<'span>,

    /// This is just there so that the `'span` lifetime is used even when both
    /// `tracing` features are disabled.
    _p: core::marker::PhantomData<&'span ()>,
}

#[derive(Debug)]
pub(crate) struct EnteredSpan {
    _enter_01: tracing::span::EnteredSpan,
}
